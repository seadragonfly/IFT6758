import os
from os.path import exists, dirname
import youtube_dl
import ffmpeg

def download_audio(YTID: str, path: str) -> None:
    """
    Download the audio of the YouTube video with ID `YTID` to `path` as MP3.
    If the destination file already exists, return without downloading.
    Handle exceptions gracefully.
    """
    # Ensure final file path ends with .mp3
    dest_mp3 = path if path.lower().endswith(".mp3") else f"{path}.mp3"

    # Skip if already present
    if exists(dest_mp3):
        return

    # Make parent directory if provided
    parent = dirname(dest_mp3)
    if parent and not exists(parent):
        os.makedirs(parent, exist_ok=True)

    url = "https://www.youtube.com/watch?v=" + YTID

    ydl_opts = {
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": dest_mp3,             # write directly to final name
        "ffmpeg_location": "/opt/homebrew/bin",  # adjust if needed
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        # "verbose": True,               # uncomment to debug
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])          # must pass a list
    except Exception as e:
        # Handle/Log the error as needed for your project
        print(f"Download failed for {YTID}: {e}")

def cut_audio(in_path: str, out_path: str, start: float, end: float) -> None:
    """
    Cut `in_path` to the [start, end] seconds range and save as `out_path`.
    """
    # Ensure output directory exists
    parent = dirname(out_path)
    if parent and not exists(parent):
        os.makedirs(parent, exist_ok=True)

    # Trim via input args (common & simple). Re-encode to mp3 to avoid frame issues.
    (
        ffmpeg
        .input(in_path, ss=start, to=end)
        .output(out_path, acodec="libmp3lame", audio_bitrate="192k")
        .overwrite_output()
        .run()
    )

# Example usage:
download_audio("L62LtChAwww", "L62LtChAwww")            # writes L62LtChAwww.mp3
cut_audio("L62LtChAwww.mp3", "trim_L62LtChAwww.mp3", 10, 20)
