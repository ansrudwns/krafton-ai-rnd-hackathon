import os
import asyncio
import subprocess
import time
import re
import itertools
from google import genai
from google.genai import types

class CoTVideoAgent:
    # Heterogeneous Multi-Model Orchestration: 가속형 Flash 모델로 시각/청각 데이터 추출, 논리형 Pro 모델로 최종 추론
    def __init__(self, map_model=None, reduce_model=None):
        self.map_model = map_model or os.environ.get("GEMINI_MAP_MODEL", "gemini-3.1-flash-lite-preview")
        self.reduce_model = reduce_model or os.environ.get("GEMINI_REDUCE_MODEL", "gemini-3.1-pro-preview")
        keys = []
        for k, v in os.environ.items():
            if k.startswith("GEMINI_API") and v.strip():
                keys.append(v.strip())
                
        if not keys:
            raise ValueError("No GEMINI_API_KEY found in environment variables.")
            
        self.num_keys = len(keys)
        print(f"[{self.num_keys} API Keys Loaded] Initializing Round-Robin API Load Balancer...")
        self.clients = [genai.Client(api_key=k) for k in keys]
        self.client_cycle = itertools.cycle(self.clients)
        # API 동시 호출 글로벌 락 (3배수 상향으로 네트워크 대역폭 극한 사용)
        self.chunk_sem = asyncio.Semaphore(self.num_keys * 3)
        # 20개 비디오 인코딩 동시 폭주(CPU 쓰래싱)를 방지하는 로컬 코어 락
        self.cpu_sem = asyncio.Semaphore(max(1, (os.cpu_count() or 4) // 2))

    def get_client(self):
        return next(self.client_cycle)

    def split_video(self, video_path: str, chunk_duration=300) -> list:
        base_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_pattern = os.path.join(base_dir, f"{base_name}_chunk_%03d.mp4")
        
        try:
            print(f"[{base_name}] Downsampling and chunking video (CPU) for instant API upload...")
            ffmpeg_exe = os.path.join(os.path.dirname(__file__), "ffmpeg_bin", "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe")
            subprocess.run([
                ffmpeg_exe, "-y", "-i", video_path,
                "-vf", "scale=640:-2,fps=1",  # 1초에 1프레임, 저해상도로 강제 압축 (Gemini 업로드 속도 극대화)
                "-b:v", "500k", "-preset", "ultrafast", # 극강의 CPU 압축 속도 보장
                "-threads", "2", # ffmpeg 단일 객체의 CPU 낭비 제한 (다중 병렬 최적화)
                "-f", "segment", "-segment_time", str(chunk_duration),
                "-reset_timestamps", "1", output_pattern
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[{base_name}] Failsafe: ffmpeg missing. Using full video.")
            return [video_path]
            
        chunks = []
        for file in sorted(os.listdir(base_dir)):
            if file.startswith(base_name + "_chunk_") and file.endswith(".mp4"):
                chunks.append(os.path.join(base_dir, file))
        return chunks if chunks else [video_path]

    def map_chunk_sync(self, chunk_path: str, prompt_text: str, chunk_idx: int) -> str:
        client = self.get_client()
        
        uploaded_file = None
        for upload_attempt in range(3):
            try:
                uploaded_file = client.files.upload(file=chunk_path)
                while uploaded_file.state.name == "PROCESSING":
                    time.sleep(2) # 5초 대기를 2초로 줄여 유휴 지연 극단적 단축
                    uploaded_file = client.files.get(name=uploaded_file.name)
                break
            except Exception as e:
                if upload_attempt == 2: raise e
                time.sleep(10)
            
        instructions = (
            "You are a strict chronological action-tracking module.\n"
            f"TARGET QUESTION: '{prompt_text}'\n\n"
            "INSTRUCTIONS:\n"
            "1. Watch the video snippet and listen to the audio carefully.\n"
            "2. Identify ONLY entities, actions, and sounds that are strictly relevant to the TARGET QUESTION.\n"
            "3. Output your findings STRICTLY in the following bulleted format.\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "- [MM:SS - MM:SS] | Entity: [Name] | Action/Sound: [Description]\n"
            "- [MM:SS - MM:SS] | Entity: [Name] | Action/Sound: [Description]\n\n"
            "If nothing relevant happens, strictly output: 'NO_RELEVANT_EVENTS'.\n"
            "Do NOT output conversational text. Do NOT answer the target question."
        )
        
        try:
            response_text = "NO_RELEVANT_EVENTS"
            # 429 Rate Limit 발생 시 자동 대기 후 재시도 (Immortal Mode)
            for inference_attempt in range(4):
                try:
                    response = client.models.generate_content(
                        model=self.map_model, # 초고속 인지 전담 모델 (Flash)
                        contents=[uploaded_file, instructions],
                        # 미디어 토큰 할당량을 최소화(프레임당 70 tokens)하여 해커톤 429 에러(Quota) 완벽 방어
                        config=types.GenerateContentConfig(temperature=0.0, media_resolution="MEDIA_RESOLUTION_LOW")
                    )
                    response_text = response.text.strip()
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if '429' in err_str or 'quota' in err_str or 'exhaust' in err_str or '503' in err_str:
                        print(f"[{os.path.basename(chunk_path)}] API Limit Hit. Safe waiting 15s... (Retrying {inference_attempt+1}/4)")
                        time.sleep(15)
                    else:
                        raise e
            return f"=== CHUNK {chunk_idx + 1} LOG ===\n{response_text}\n"
        finally:
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except:
                    pass

    def reduce_solve_sync(self, full_log: str, prompt_text: str) -> str:
        client = self.get_client()
        instructions = (
            "You are the master Logical Reasoner. You must solve a multiple-choice question strictly based on the provided chronological video logs.\n\n"
            "INSTRUCTIONS:\n"
            "STEP 1: Review the combined chunk logs thoroughly.\n"
            "STEP 2: Write out your step-by-step reasoning. You must aggressively use the 'Process of Elimination'.\n"
            "STEP 3: Evaluate every single multiple-choice option explicitly. If the log lacks direct positive evidence, categorically rule out contradictory or medically/physically absurd options.\n"
            "STEP 4: You must ALWAYS provide a final choice. If you are still unsure after elimination, you MUST make the most plausible EDUCATED GUESS out of the remaining valid letters in the prompt. Do not abstain.\n"
            "STEP 5: Output EXACTLY ONE single English letter corresponding to your choice inside the XML block. Example: <ANSWER>A</ANSWER>. Do NOT write words like '<ANSWER>Option A</ANSWER>'."
        )
        context = f"MASTER TIMELINE LOGS:\n{full_log}\n\nTARGET QUESTION:\n{prompt_text}"
        
        response_text = "X"
        for inference_attempt in range(4):
            try:
                response = client.models.generate_content(
                    model=self.reduce_model, # 통찰력 기반 메인 논리 전담 모델 (Pro)
                    contents=[context],
                    config=types.GenerateContentConfig(system_instruction=instructions, temperature=0.0)
                )
                response_text = response.text.strip()
                break
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or 'quota' in err_str or 'exhaust' in err_str or '503' in err_str:
                    print(f"[Reasoner] API Limit Hit. Safe waiting 15s... (Retrying {inference_attempt+1}/4)")
                    time.sleep(15)
                else:
                    raise e
        
        # 정답 추출 무결성 강화 (XML 태그 오류 방지)
        match = re.search(r'<ANSWER>\s*([A-Z])\s*</ANSWER>', response_text.upper())
        if match: 
            return match.group(1)
            
        # Fallback 1: 보기(Option)와 관련된 명확한 키워드 근처의 단일 문자를 추출
        fallback_match = re.search(r'(?:ANSWER|OPTION|CHOICE)[^A-Z]*([A-Z])\b', response_text.upper())
        if fallback_match:
            return fallback_match.group(1)
            
        return 'X'

    async def run(self, video_path: str, prompt_text: str) -> str:
        try:
            # CPU 집중 작업(인코딩)은 코어 락으로 묶어 콘텍스트 스위칭 오버헤드 제거
            async with self.cpu_sem:
                chunks = await asyncio.to_thread(self.split_video, video_path)
            
            async def process_chunk(chunk_path, idx):
                async with self.chunk_sem:
                    try:
                        log_txt = await asyncio.to_thread(self.map_chunk_sync, chunk_path, prompt_text, idx)
                        return idx, log_txt
                    finally:
                        if chunk_path != video_path and os.path.exists(chunk_path):
                            try:
                                os.remove(chunk_path)
                            except:
                                pass
                
            chunk_tasks = [process_chunk(c, i) for i, c in enumerate(chunks)]
            results = await asyncio.gather(*chunk_tasks)
            results.sort(key=lambda x: x[0])
            
            master_log = "\n\n".join(txt for _, txt in results)
            print(f"[{video_path}] All chunks processed. Running Map-Reduce Final Logic...")
            answer = await asyncio.to_thread(self.reduce_solve_sync, master_log, prompt_text)
            
            return answer
        except Exception as e:
            print(f"[{video_path}] Fatal Pipeline Error: {e}")
            return 'X'
