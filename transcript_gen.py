from yt_dlp import YoutubeDL
from pydub import AudioSegment
import os

from deepgram import DeepgramClient, PrerecordedOptions

DEEPGRAM_API_KEY = 'deepgram-key'

# download audio
def download_audio(url, output_format="mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio/downloaded_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3' if output_format == "mp3" else None,
            'preferredquality': '192',
        }],
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Convert to desired format if not mp3
    output_file = "audio/downloaded_audio.mp3"  # Default to mp3
    if output_format != "mp3":
        audio = AudioSegment.from_file("audio/downloaded_audio.mp3", format="mp3")
        output_file = f"audio/downloaded_audio.{output_format}"
        audio.export(output_file, format=output_format)
        os.remove("downloaded_audio.mp3")  # Remove temporary mp3 file if converted

    return output_file



def transcript_gen(url):
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)

    output_path = download_audio(url)

    with open(output_path, 'rb') as buffer_data:
        payload = { 'buffer': buffer_data }

        options = PrerecordedOptions(
            smart_format=True, model="nova-2", language="en-US"
        )

        response = deepgram.listen.rest.v('1').transcribe_file(payload, options)
        return response



def main():
    print('This is your transcript_gen script')

if __name__ == '__main__':
    main()
    transcript_gen('https://www.youtube.com/watch?v=ayKJsoa-GU0')