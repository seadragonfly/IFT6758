import re
import os
import pandas as pd
from tqdm import tqdm
from q2 import download_audio, cut_audio
from q1 import contains_label
from typing import List


def filter_df(csv_path: str, label: str) -> List[str]:
    df= pd.read_csv(csv_path)
    matches = contains_label(df["label_names"], label)
    mask = df["label_names"].isin(matches)
    
    return df[mask]


def data_pipeline(csv_path: str, label: str) -> None:
    df=filter_df(csv_path, label)


    raw_dir = os.path.join(f"{label}_raw")
    cut_dir = os.path.join(f"{label}_cut")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cut_dir, exist_ok=True)


    for i in tqdm(range(len(df)), total=len(df), desc=f"{label}"):
        row = df.iloc[i]
        ytid= row["# YTID"]
        d_path   = os.path.join(raw_dir, f"{ytid}.mp3")
        out_path = os.path.join(cut_dir, f"{ytid}.mp3")
        start=float(row[" start_seconds"])
        end=float(row[" end_seconds"])

        try:
            download_audio(ytid, d_path)
            cut_audio(d_path, out_path, start, end)
        except Exception as e:
            print(f"[WARN] {ytid}: {e}")
            continue
    pass

def rename_files(path_cut: str, csv_path: str) -> None:

    df= pd.read_csv(csv_path)
    folder = path_cut
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for i in files:
        old_name=i
        m = re.match(r"^(.*)\.mp3$", i)
        ytid_file = m.group(1)
            
        for i in range(len(df)):
            row = df.iloc[i]
            ytid_df=row["# YTID"]
            if ytid_df==ytid_file:
                start=int(row[" start_seconds"])
                end=int(row[" end_seconds"])
                dif=int(end-start)
                new_name=f'{ytid_file}_{str(start)}_{str(end)}_{str(dif)}.mp3'
                os.rename(folder+"/"+old_name, folder+"/"+new_name)


    # TODO
    pass


if __name__ == "__main__":
    print(filter_df("audio_segments_clean.csv", "Laughter"))
    data_pipeline("audio_segments_clean.csv", "Laughter")
    rename_files("Laughter_cut", "audio_segments_clean.csv")
