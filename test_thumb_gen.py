import numpy as np
import cv2
from yt_dlp import YoutubeDL

def download_youtube_video(video_url, output_path="audio/download_video2.mp4"):
    ydl_opts_info = {
        'format': 'worst',  # You can adjust this to specify format options
        'noplaylist': True,  # Ensure we're only downloading the video, not a playlist
        'quiet': True  # Suppress the download output
    }

    with YoutubeDL(ydl_opts_info) as ydl:
        # Extract video information
        info_dict = ydl.extract_info(video_url, download=False)
    # print(info_dict)
    # return info_dict
    # Define options for yt-dlp
    ydl_opts_vid = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',  # Force MP4 format
        'outtmpl': output_path if output_path else '%(title)s.%(ext)s',  # Specify output path and file name
        'merge_output_format': 'mp4',
    }

    # Use yt-dlp to download the video
    with YoutubeDL(ydl_opts_vid) as ydl:
        ydl.download([video_url])

    output_file = "audio/downloaded_video.mp4" 
    return [output_file, info_dict]

def extract_frames(video_path, frame_interval=60):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))

        frame_count += 1

    cap.release()
    return frames


def calculate_color_diversity(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixels = rgb_frame.reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    return unique_colors

def select_frame_with_highest_color_diversity(frames):
    best_frame = None
    max_color_diversity = 0

    for frame_count, frame in frames:
        diversity = calculate_color_diversity(frame)

        if diversity > max_color_diversity:
            max_color_diversity = diversity
            best_frame = (frame_count, frame)

        print(diversity)
    return best_frame

def save_thumbnail(frame, output_path):
    cv2.imwrite(output_path, frame)
    print(f"Thumbnail saved to {output_path}")

def main():
    # data = download_youtube_video('https://www.youtube.com/watch?v=g-x6tQYwl84')
    video_path = 'audio/download_video1.mp4'
    output_path = 'audio/thumbnail1.jpg'
    frames = extract_frames(video_path)

    frame_count, best_frame = select_frame_with_highest_color_diversity(frames)
    save_thumbnail(best_frame, output_path)

if __name__ == '__main__':
    main()