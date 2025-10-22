import youtube_dl
import ffmpeg
import pandas as pd
import numpy as np
import csv
import threading
from tqdm import tqdm
from os.path import exists
import yt_dlp
import os

def download_audio(YTID: str, path: str) -> None:
    if exists(path):
        return None
    
    url='https://www.youtube.com/watch?v='+YTID
    stem, _ = os.path.splitext(path)             
    ydl_opts = {
        "format": "bestaudio/best",
        #"ffmpeg_location": ffmpeg_path,
        "outtmpl": stem,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "overwrites": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f'Downloading {url}')
            ydl.download([url])
        except Exception as e:
            print(f"[WARN] {url}: {e}")


def cut_audio(in_path: str, out_path: str, start: float, end: float) -> None:
    file = ffmpeg.input(in_path)
    trimmed = file.filter('atrim', start=start, end=end)
    output = ffmpeg.output(trimmed, out_path)
    output.run(overwrite_output=True)
    print(f"Cut successful: {out_path}")
    pass