import os
import asyncio
import traceback
import time
from typing import List, Tuple
from video_processor import compress_and_chunk_video
from agent_pipeline import process_single_video_pipeline
import glob

async def worker_task(
    index: int, 
    video_path: str, 
    prompt_path: str, 
    semaphore: asyncio.Semaphore,
    temp_dir: str,
    shared_results: List[str],
    file_lock: asyncio.Lock
) -> Tuple[int, str]:
    """동시성 제어(Semaphore) 아래 개별 비디오 처리"""
    async with semaphore:
        print(f"[Task {index}] Started for {os.path.basename(video_path)}")
        
        with open(prompt_path, 'r', encoding='utf-8', errors='ignore') as f:
            prompt_text = f.read()
            
        compressed_path = os.path.join(temp_dir, f"compressed_{index}.mp4")
        answer = "X"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 1. 초고속 압축 및 청킹
                print(f"[Task {index}] Compressing & Chunking video (Attempt {attempt+1})...")
                chunks = await compress_and_chunk_video(video_path, temp_dir, f"video{index}")
                
                # 2. 파이프라인 처리
                print(f"[Task {index}] Running Agent Pipeline with {len(chunks)} chunks...")
                answer = await process_single_video_pipeline(chunks, prompt_text)
                print(f"[Task {index}] Finished. Answer: {answer}")
                break # 성공 시 루프 탈출
            except Exception as e:
                print(f"[Task {index}] Failed with error:")
                traceback.print_exc()
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = 4 ** attempt
                    print(f"[Task {index}] Rate limit hit. Backing off for {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"[Task {index}] Unrecoverable error.")
                    break
            finally:
                # 중간 파일 정리 (모든 청크 삭제)
                search_pattern = os.path.join(temp_dir, f"video{index}_chunk_*.mp4")
                for path in glob.glob(search_pattern):
                    try:
                        os.remove(path)
                    except:
                        pass

        # 실시간 부분 결과 백업 저장 (Incremental Saving)
        async with file_lock:
            shared_results[index - 1] = answer
            current_ans = "".join(shared_results)
            with open("submission.txt", "w", encoding='utf-8') as f:
                f.write(current_ans)
            print(f"[Backup] Saved intermediate submission: {current_ans}")
                        
        return index, answer

async def main():
    start_time = time.time()
    test_folder = "target_videos"
    temp_dir = "temp_compressed"
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    pairs = []
    # Hackathon spec: 20 videos
    for i in range(1, 21):
        vid = os.path.join(test_folder, f"video{i}.mp4")
        prm = os.path.join(test_folder, f"prompt{i}.txt")
        if os.path.exists(vid) and os.path.exists(prm):
            pairs.append((i, vid, prm))
            
    if not pairs:
        print(f"[-] No video/prompt pairs found in '{test_folder}'.")
        print("Please place video1.mp4, prompt1.txt, etc., inside the folder and run again.")
        return

    print(f"[+] Found {len(pairs)} pairs to process.")
    
    # 동시 스레드 수 제어 (4~5개가 안전)
    semaphore = asyncio.Semaphore(5)
    file_lock = asyncio.Lock()
    shared_results = ["X"] * 20
    
    tasks = []
    for num, v, p in pairs:
        tasks.append(asyncio.create_task(worker_task(num, v, p, semaphore, temp_dir, shared_results, file_lock)))
        
    results = await asyncio.gather(*tasks)
    
    # 인덱스(번호) 순으로 정렬
    results.sort(key=lambda x: x[0])
    
    final_string = "".join([r[1] for r in results])
    print("\n" + "="*50)
    print(f"[FINAL RESULT] {final_string}")
    print("="*50)
    
    with open("submission.txt", "w", encoding='utf-8') as f:
        f.write(final_string)
        
    elapsed = time.time() - start_time
    print(f"Saved result to submission.txt (Total Time: {elapsed:.2f} seconds)")

if __name__ == "__main__":
    asyncio.run(main())
