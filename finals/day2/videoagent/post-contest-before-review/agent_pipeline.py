import os
import re
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env 파일 로드
load_dotenv()

# Vertex AI Express API Key가 있지만 SDK 충돌 이슈를 피하기 위해 
api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(
    vertexai=True,
    api_key=api_key
)

def extract_answer_letter(text: str) -> str:
    """
    주어진 텍스트의 끝부분이나 전체에서 정답 알파벳 1개를 추출합니다.
    (V3의 강력한 XML 기반 태그 추출 채택)
    """
    # 1. XML 태그 추출 시도 (강제력 부여된 Output)
    match = re.search(r'<ANSWER>\s*([A-Z])\s*</ANSWER>', text.upper())
    if match: return match.group(1)
    
    # 2. 마지막의 결론 부분을 차선책으로 탐색
    matches = re.findall(r'([A-Z])(?:\)|\s*$|\.$)', text.strip())
    if matches:
        return matches[-1].upper()
    
    # 3. 실패 시 단일 대문자 탐색 (알파벳 단독)
    match = re.search(r'\b([A-Z])\b(?![A-Za-z])', text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    return "X"
def extract_options(prompt_text: str) -> list[str]:
    """프롬프트에서 객관식 보기 텍스트를 추출하여 워커를 편향시키기 위해 사용합니다."""
    options = []
    for line in prompt_text.split('\n'):
        line = line.strip()
        # "A) 11", "B. 14", "C: 22" 형태 추출
        match = re.match(r'^[A-Z][\)\.:]\s*(.+)', line)
        if match:
            options.append(match.group(1).strip())
    return options

chunk_semaphore = asyncio.Semaphore(3)

async def run_worker_map_phase(chunk_path: str, prompt_text: str, chunk_index: int) -> str:
    """
    Map Phase (Worker) - Use Inline Data for Vertex AI with Rate Pacing
    """
    async with chunk_semaphore:
        max_retries = 4
        for attempt in range(max_retries):
            print(f"[Worker] Reading {chunk_path} (Attempt {attempt+1})...")
            try:
                with open(chunk_path, 'rb') as f:
                    video_data = f.read()
                    
                print(f"[Worker] Processing {os.path.basename(chunk_path)} ({len(video_data)} bytes)...")
                options = extract_options(prompt_text)
                spoiler_text = ""
                if options:
                    spoiler_text = "\n4. TARGET BIAS: The multiple choice options include: " + ", ".join(options) + ". You MUST highly prioritize searching for counts, objects, or actions that align with these specific thresholds. Ignore numbers completely out of this range."
                    
                prompt = (
                    "You are a strict chronological action-tracking module.\n"
                    f"TARGET QUESTION: '{prompt_text}'\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Watch the video and listen to the audio carefully.\n"
                    "2. Identify ONLY entities, actions, and sounds that are strictly relevant to the TARGET QUESTION. Aggressively ignore anything irrelevant.\n"
                    "3. Output your findings STRICTLY in the following bulleted format.\n"
                    f"{spoiler_text}\n\n"
                    "REQUIRED OUTPUT FORMAT:\n"
                    "- [MM:SS - MM:SS] | Entity: [Name] | Action/Sound: [Description]\n"
                    "If nothing relevant happens, strictly output: 'NO_RELEVANT_EVENTS'.\n"
                    "Do NOT output conversational text. Do NOT answer the target question."
                )
                
                video_part = types.Part.from_bytes(
                    data=video_data,
                    mime_type="video/mp4"
                )
                
                response = await client.aio.models.generate_content(
                    model='gemini-3.1-flash-lite-preview',
                    contents=[video_part, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        media_resolution="MEDIA_RESOLUTION_LOW"
                    )
                )
                return f"=== CHUNK {chunk_index} LOG ===\n{response.text}\n"
            except Exception as e:
                print(f"[Worker] Error processing {chunk_path}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[Worker] Retrying {os.path.basename(chunk_path)} in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    return f"=== CHUNK {chunk_index} LOG ===\nNO_RELEVANT_EVENTS\n"

async def run_master_reduce_phase(logs: str, prompt_text: str) -> str:
    """
    Reduce Phase (Master) with Self-Reflection
    """
    print("[Master] Reasoning initial answer based on logs...")
    # Step 1: Initial Reasoning
    prompt_step1 = (
        f"MASTER TIMELINE LOGS:\n{logs}\n\nTARGET QUESTION:\n{prompt_text}\n\n"
        "You are the master Logical Reasoner. You must solve a multiple-choice question strictly based on the provided chronological video logs.\n\n"
        "INSTRUCTIONS:\n"
        "STEP 1: Review the combined logs thoroughly.\n"
        "STEP 2: Write out your step-by-step reasoning. You must aggressively use the 'Process of Elimination'.\n"
        "STEP 3: Evaluate every single multiple-choice option explicitly.\n"
        "STEP 4: Conclude with a tentative answer, but do NOT use the <ANSWER> tag yet."
    )
    
    try:
        response_step1 = await client.aio.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt_step1,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        initial_reasoning = response_step1.text
    except Exception as e:
        print(f"[Master] Error during step 1: {e}")
        return "X"
        
    print(f"[Master] Running Self-Reflection...")
    # Step 2: Self-Reflection & Final Output
    prompt_step2 = (
        f"TARGET QUESTION:\n{prompt_text}\n\n"
        f"YOUR PREVIOUS REASONING:\n{initial_reasoning}\n\n"
        "INSTRUCTION:\n"
        "1. Act as a harsh critic and review your own reasoning above.\n"
        "2. Check if there are any math calculation errors, logical fallacies, or double-counting in your logic.\n"
        "3. Make any necessary corrections and state your final verified logic.\n"
        "4. You must ALWAYS provide a final choice. If unsure, make the most plausible EDUCATED GUESS out of the valid letters.\n"
        "5. Output EXACTLY ONE single English letter corresponding to your FINAL choice inside the XML block. Example: <ANSWER>A</ANSWER>. Do NOT write words like '<ANSWER>Option A</ANSWER>'."
    )
    
    try:
        response_step2 = await client.aio.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt_step2,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        final_text = response_step2.text
        print(f"[Master] Final Result: {final_text}")
        return extract_answer_letter(final_text)
    except Exception as e:
        print(f"[Master] Error during reflection: {e}")
        return extract_answer_letter(initial_reasoning) # Fallback to initial if reflection fails

async def process_single_video_pipeline(video_chunks: list[str], prompt_text: str) -> str:
    """조각난 비디오 청크들을 병렬로 처리하는 개선된 파이프라인"""
    # 1. 병렬 Map Phase (Worker)
    map_tasks = [run_worker_map_phase(chunk, prompt_text, i) for i, chunk in enumerate(video_chunks)]
    logs_array = await asyncio.gather(*map_tasks)
    
    # 순서대로 병합
    master_log = "\n\n".join(logs_array)
    
    # 2. Reduce Phase (Master)
    answer = await run_master_reduce_phase(master_log, prompt_text)
    return answer
