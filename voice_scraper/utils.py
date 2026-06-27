
from argparse import Namespace
import os

from numpy import dtype, ndarray
import numpy as np
from numpy.typing import NDArray
from voice_scraper.embedding import get_embedding, cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.neighbors import NearestNeighbors
from voice_scraper.models import SegmentData, Clip
from pydantic import BaseModel
from typing import Any, cast
from rich import print
import ffmpeg

def resolve_path() -> str:
    output_dir = os.environ.get("OUTPUT_DIR")
    folder_path = ""
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        folder_path = output_dir
    else:
        import tempfile
        temp_dir = tempfile.gettempdir()
        folder_path = os.path.join(temp_dir, "voice_scraper_downloads")
        os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def resolve_duration(duration: str) -> int:
    minutes, seconds = map(int, duration.split(":"))
    return minutes * 60 + seconds

def check_path_exists(path: str) -> bool:
    return os.path.exists(path)


def join_segments(segments: list[str], output_path: str) -> None:
    tracks = [ffmpeg.input(segment) for segment in segments]
    ffmpeg.concat(*tracks, v=0, a=1).output(output_path).run(quiet=True, overwrite_output=True)


def join_segmented_clips(segment_data: SegmentData, segment_folder: str, clip: Clip)->list[str]:
    speaker_paths = []
    for speaker in range(segment_data.number_of_speakers):
        req_segments = []
        req = []
        for segment in segment_data.segments:
            if segment.speaker_label == speaker:
                req_segments.append(segment)
                req.append(os.path.join(segment_folder, f"{segment.id}.wav"))
        time_taken = req_segments[-1].end_time - req_segments[0].start_time
        if time_taken < 0.05:
            print(f"[yellow]Skipping speaker_'{speaker}' for '{clip.name}'[/yellow]")
            print(f"[yellow] Too short ({time_taken} seconds)[/yellow]")
            continue
        speaker_path = os.path.join(os.path.dirname(clip.path), f"{clip.id}_speaker_{speaker}.wav")
        join_segments(req, speaker_path)
        speaker_paths.append(speaker_path)
    return speaker_paths

def ensure_sample_voice(args: Namespace) -> bool:
    if args.sample_voice is None:
        return True
    if not os.path.exists(args.sample_voice):
        print(f"[red]Error: Sample voice '{args.sample_voice}' does not exist.[/red]")
        return False
    if str(args.sample_voice).endswith(".wav"):
        print(f"[red]Error: Sample voice '{args.sample_voice}' is not a .wav file.[/red]")
        return False
    return True
