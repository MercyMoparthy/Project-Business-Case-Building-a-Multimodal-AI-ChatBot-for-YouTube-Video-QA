import os
import pandas as pd

def load_metadata(csv_path):
    return pd.read_csv(csv_path)

def load_transcript(video_id, transcript_dir):
    filepath = os.path.join(transcript_dir, f"{video_id}.txt")
    with open(filepath, 'r') as f:
        return f.read()