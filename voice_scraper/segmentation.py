from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from dotenv import load_dotenv
import os
import time
import ffmpeg
from datetime import datetime
from voice_scraper.models import Clip, Segment, SegmentData
load_dotenv()

def generate_clip_id(character: str) -> str:
    return f"{character.replace(' ', '_')}_clip_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


def load_model() -> Pipeline:
    try:
        pipeline = Pipeline.from_pretrained(
            os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization"),
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        if pipeline is not None:
            return pipeline
        else:
            raise RuntimeError("Failed to load pipeline")
    except Exception as e:
       raise e

def run_diarization(clip: Clip) -> SegmentData:
    try:
        pipeline = load_model()
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit(1)

    with ProgressHook() as hook:
        output = pipeline(clip.path, hook=hook) 

    segments: list[Segment] = []
    speakers: dict[str, int] = {}
    for turn, speaker in output.speaker_diarization:

        if speaker not in speakers:
            speakers[speaker] = len(speakers)

        segments.append(Segment(
            id=generate_clip_id(clip.name),
            clip_id=clip.id,
            start_time=turn.start,
            end_time=turn.end,
            speaker_label=speakers[speaker]
        ))

    return SegmentData(
        id=generate_clip_id(clip.name),
        clip_id=clip.id,
        number_of_speakers=len(speakers),
        audio_path=clip.path,
        segments=segments
    )

def run_segmentation(segment_data: SegmentData, output_path: str) -> list[str]:
    segments = segment_data.segments
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_path, f"{segments[i].id}.wav")
        (
            ffmpeg
            .input(segment_data.audio_path, ss=segment.start_time, to=segment.end_time)
            .output(output_file)
            .run(quiet=True, overwrite_output=True)
        )
    
    return [os.path.join(output_path, f"{segment.id}.wav") for segment in segments]

import argparse
args = argparse.ArgumentParser()
args.add_argument("--audio-path", type=str, required=True, help="Path to the audio file to be processed")
args.add_argument("--output-path", type=str, required=True, help="Directory where the segmented audio files will be saved")


if __name__ == "__main__":
    from pprint import pprint
    parsed_args = args.parse_args()

    clip = Clip(
        id=generate_clip_id("test_character"),
        name="test_character",
        path=parsed_args.audio_path,
        duration=212
    )

    segment_data = run_diarization(clip)
    for segment in segment_data.segments:
        print(segment)
    output_files = run_segmentation(segment_data, parsed_args.output_path)
    print(f"Generated segment files: {output_files}")