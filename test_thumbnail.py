import streamlit as st
import openai
from yt_dlp import YoutubeDL


def download_youtube_video(video_url, output_path="audio/download_video.mp4"):
    ydl_opts_info = {
        'format': 'best',  # You can adjust this to specify format options
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
    return output_file, info_dict


# Set up your OpenAI API key
client = openai.OpenAI(api_key='api_key')

def generate_summary(prompt):
    prompt = "Generate a summary by analyzing the following transcript. Make sure to generate a concise summary:" + prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    summary = response.choices[0].message.content
    # print(summary)
    return summary

def generate_caption(prompt):
    prompt = "Generate a caption based on the following text:" + prompt

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{"role":"system", "content":"You are a advertisement agent for youtubers.Your task is to create eye catching phrases."},
                    {"role": "user", "content": prompt}]
    )

    caption = response.choices[0].message.content
    print(caption)
    return caption

# Function to generate an image using OpenAI's new API method
def generate_image(prompt):
    prompt = "Generate a list of words which make up the core idea of the following text. Only return word 1, word 2, word 3 etc.:" + prompt

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{"role": "user", "content": prompt}]
    )

    caption = response.choices[0].message.content
    print(caption)

    prompt = "Generate an eye catching thumbnail image based on the following text. Image should contain the objects as given. Do not add any text in the image itself :" + caption
    response = client.images.generate(
    prompt=prompt,
    n=2,
    size="512x512" # 1280 x 720
    )

    image_url = response.data[0].url
    print(image_url)
    return image_url


def main():
    print('THis is thumnail file')
    # video_url = 'https://www.youtube.com/watch?v=ayKJsoa-GU0'
    # video_info = download_youtube_video(video_url)
    # print(f"Title: {video_info.get('title')}")
    # print(f"Uploader: {video_info.get('uploader')}")
    # print(f"Upload Date: {video_info.get('upload_date')}")
    # print(f"Duration: {video_info.get('duration')} seconds")
    # print(f"View Count: {video_info.get('view_count')}")
    # print(f"Like Count: {video_info.get('like_count')}")
    # print(f"Description: {video_info.get('description')}")
    # print(f"Tags: {video_info.get('tags')}")
if __name__=="__main__":
    main()