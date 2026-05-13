import os
import argparse
import asyncio
import concurrent.futures
import re
from agent_v3 import CoTVideoAgent
from dotenv import load_dotenv

def find_target_files(test_dir, index):
    idx_str = str(index)
    idx_pad = f"{index:02d}"
    video_exts = ('.mp4', '.mkv', '.avi', '.mov', '.webm')
    text_exts = ('.txt', '.json', '.md')
    
    all_files = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            all_files.append(os.path.join(root, f))
            
    def matches_idx(fpath, idx, pad):
        base = os.path.basename(fpath)
        parent = os.path.basename(os.path.dirname(fpath))
        if parent == idx or parent == pad: return True
        if re.search(rf'(?<!\d){idx}(?!\d)', base) or re.search(rf'(?<!\d){pad}(?!\d)', base): return True
        return False

    v_cands = [f for f in all_files if f.lower().endswith(video_exts) and matches_idx(f, idx_str, idx_pad)]
    t_cands = [f for f in all_files if f.lower().endswith(text_exts) and matches_idx(f, idx_str, idx_pad)]
    
    return v_cands[0] if v_cands else None, t_cands[0] if t_cands else None

async def process_pair(agent: CoTVideoAgent, i: int, test_dir: str, sem: asyncio.Semaphore, shared_results: list, file_lock: asyncio.Lock):
    async with sem:
        video_path, prompt_path = await asyncio.to_thread(find_target_files, test_dir, i)
        
        if not video_path or not prompt_path or not os.path.exists(video_path) or not os.path.exists(prompt_path):
            print(f"[Warn] Missing files for index {i}")
            answer = "X"
        else:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            answer = await agent.run(video_path, prompt_text)
            
        print(f"[Finished {i}] Final LLM Decision -> {answer}")
        
        async with file_lock:
            shared_results[i-1] = answer
            current_ans = "".join(shared_results)
            with open("submission_v3.txt", "w") as f:
                f.write(current_ans)
            print(f"[Backup] Saved intermediate submission: {current_ans}")
            
        return i, answer

async def main():
    # 파이썬 기본 ThreadPool 제한(CPU 코어 수 에 종속)을 강제 해제하여 API 통신 네트워크 병렬성 극대화
    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=100))
    
    # 명시적으로 VideoAgent 폴더 내부의 .env를 계속 사용하도록 연결
    env_path = os.path.join(os.path.dirname(__file__), "..", "VideoAgent", ".env")
    load_dotenv(env_path)
    parser = argparse.ArgumentParser(description="V3 API-Only Contingency Runner")
    parser.add_argument("test_folder", help="Path to test folder")
    parser.add_argument("--num_videos", type=int, default=20)
    args = parser.parse_args()
    
    try:
        agent = CoTVideoAgent()
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # API 키 개수에 비례하여 동시 접속량(Semaphore)을 무한 확장
    # 무료 티어 제한(통상 15 RPM)을 절대 넘지 않기 위해 1개 키당 3대(12 청크)만 병렬로 돌려 안전을 극대화합니다.
    dynamic_concurrency = agent.num_keys * 3
    sem = asyncio.Semaphore(dynamic_concurrency)
    
    file_lock = asyncio.Lock()
    shared_results = ["X"] * args.num_videos
    
    print(f"*** TRIGGERING V3 API-ONLY FALLBACK ***")
    print(f"Detected {agent.num_keys} API keys. Scaling concurrency limit to {dynamic_concurrency} videos simultaneously.")
    
    tasks = []
    for i in range(1, args.num_videos + 1):
        tasks.append(process_pair(agent, i, args.test_folder, sem, shared_results, file_lock))
        
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])
    
    final_string = "".join(ans for _, ans in results)
    print("=" * 40)
    print(f"V3 FINAL ANSWER STRING:\n{final_string}")
    print("=" * 40)
    
    with open("submission_v3.txt", "w") as f:
        f.write(final_string)
    print("Answer exported to submission_v3.txt")

if __name__ == "__main__":
    asyncio.run(main())
