# Create folder with id 
# save the audio file in that folder
# then do whatever next

from importlib.resources import path

import ffmpeg

from voice_scraper.models import Clip, SearchResults
import yt_dlp
import os
from dotenv import load_dotenv

from voice_scraper.search import generate_id
from voice_scraper.utils import resolve_path
load_dotenv()

def convert_to_wav(input_file: str, output_file: str) -> None:
    (
        ffmpeg
        .input(input_file)
        .output(output_file, format="wav", acodec="pcm_s16le", ac=1, ar="16k")
        .run(quiet=True, overwrite_output=True)
    )


def download_audio(search_result: SearchResults, folder_path: str | None = None) -> Clip | None:
    if folder_path is None:
        folder_path = resolve_path()

    ydl = yt_dlp.YoutubeDL({
        "format": "bestaudio/best",
        "outtmpl": os.path.join(folder_path, f"{search_result.id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    })
    try:
        info = ydl.extract_info(search_result.url, download=True)
        convert_to_wav(os.path.join(folder_path, f"{search_result.id}.mp3"), os.path.join(folder_path, f"{search_result.id}.wav"))
        os.remove(os.path.join(folder_path, f"{search_result.id}.mp3"))
    except Exception as e:
        print(f"[red]Error downloading audio for '{search_result.name}': {e}[/red]")
        return None
    
    return Clip(
        id=search_result.id,
        name=search_result.name,
        path=os.path.join(folder_path, f"{search_result.id}.wav"),
        duration=search_result.duration
    )

def get_yt_video_info(url: str, name: str) -> SearchResults:
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return SearchResults(
        id=generate_id(name),
        url=url,
        duration=info.get("duration") or None,
        name=name,
        content=info.get("title") or f"{name} audio",
    )

if __name__ == "__main__":
    search_result = SearchResults(
        id="watkins",
        url="https://www.youtube.com/shorts/oFbKmSZRgiU",
        duration=212,
        name="Test Character",
        content="IDK"
    )
    audio_file = download_audio(search_result)
    print(f"Downloaded audio file: {audio_file}")