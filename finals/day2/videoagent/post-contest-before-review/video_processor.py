import asyncio
import os
import glob
from typing import List

async def compress_and_chunk_video(input_path: str, temp_dir: str, base_name: str) -> List[str]:
    """
    Compresses video to 2fps and 720px width using FFmpeg (ultrafast preset)
    and chunks it into 5-minute segments while preserving absolute timestamps.
    Returns a sorted list of chunk paths.
    """
    FFMPEG_PATH = r"C:\Users\m8686\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
    
    output_pattern = os.path.join(temp_dir, f"{base_name}_chunk_%03d.mp4")
    
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-vf", "scale=720:-2,fps=2",
        "-c:v", "libx264",
        "-crf", "28",
        "-preset", "ultrafast",
        "-an",
        "-f", "segment",
        "-segment_time", "300",
        "-reset_timestamps", "0",
        output_pattern
    ]
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        err_msg = stderr.decode('utf-8', errors='ignore')
        print(f"[-] FFmpeg error for {input_path}:\n{err_msg}")
        raise RuntimeError(f"FFmpeg compression failed for {input_path}")
        
    # Gather chunks
    search_pattern = os.path.join(temp_dir, f"{base_name}_chunk_*.mp4")
    chunks = sorted(glob.glob(search_pattern))
    
    if not chunks:
        raise RuntimeError(f"FFmpeg produced no chunks for {input_path}")
        
    return chunks
