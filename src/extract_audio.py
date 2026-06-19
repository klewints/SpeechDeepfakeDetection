import os
import pandas as pd
from moviepy import VideoFileClip

csv_path = "dataset/raw/HAV-DF/video_metadata.csv"

train_video_folder = "dataset/raw/HAV-DF/train_videos"

test_video_folder = "dataset/raw/HAV-DF/test_videos"

output_folder = "dataset/audio"

# Create output folder
os.makedirs(output_folder, exist_ok=True)


df = pd.read_csv(csv_path)

print("Total rows:", len(df))


for idx, row in df.iterrows():

    video_name = row["video_name"]

    label = row["label"].lower()

    # Determine whether video is in train or test
    train_path = os.path.join(
        train_video_folder,
        video_name
    )

    test_path = os.path.join(
        test_video_folder,
        video_name
    )

    if os.path.exists(train_path):
        video_path = train_path

    elif os.path.exists(test_path):
        video_path = test_path

    else:
        print(f"Video not found: {video_name}")
        continue

    # Create label folder
    label_folder = os.path.join(
        output_folder,
        label
    )

    os.makedirs(label_folder, exist_ok=True)

    # Output wav filename
    audio_name = video_name.replace(".mp4", ".wav")

    audio_path = os.path.join(
        label_folder,
        audio_name
    )

    # Skip if already extracted
    if os.path.exists(audio_path):
        print("Already exists:", audio_name)
        continue

    try:

        clip = VideoFileClip(video_path)

        clip.audio.write_audiofile(
            audio_path,
            fps=16000,
            logger=None
        )

        print("Done:", audio_name)

    except Exception as e:

        print("Error:", video_name)
        print(e)