import argparse
import logging
import warnings
import os

import ffmpeg
from voice_scraper.search import search_voice_samples
from voice_scraper.downloader import download_audio, get_yt_video_info
from voice_scraper.segmentation import generate_clip_id, run_diarization, run_segmentation
from rich.progress import track
import voice_scraper.console as console
from voice_scraper.models import Clip, Segment, SegmentData
from rich import print
from rich_argparse import RawDescriptionRichHelpFormatter
from voice_scraper.speaker_identifier import find_speaker_character_3, find_speaker_character_with_sample
from voice_scraper.utils import resolve_path, resolve_duration, join_segmented_clips, join_segments, ensure_sample_voice
from textwrap import dedent


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)



RawDescriptionRichHelpFormatter.styles["argparse.args"] = "bold green"
RawDescriptionRichHelpFormatter.styles["argparse.groups"] = "bold magenta"
RawDescriptionRichHelpFormatter.styles["argparse.prog"] = "bold magenta"
RawDescriptionRichHelpFormatter.styles["argparse.help"] = "white"

parser = argparse.ArgumentParser(
    description="Scrape voice samples from the web by a query.",
    epilog=dedent("""
Example usage:
    uv run voice_scraper/cli.py -c "Frieren" -q dub english voice -l 5 --max-duration 5:00 --use-yt-search

    uv run voice_scraper/cli.py -c "Frieren"
"""),
    formatter_class=RawDescriptionRichHelpFormatter
)

parser.add_argument(
    "-q", "--query",
    nargs="+",
    default=["talking", "speaking", "podcast"],
    required=False,
    help="Search query for voice samples"
)
parser.add_argument(
    "-c", "--character",
    type=str,
    required=True,
    help="Character name for the voice sample"
)
parser.add_argument(
    "--url",
    type=str,
    required=False,
    help="Direct URL to the voice sample (overrides search query)"
)

parser.add_argument(
    "--use-yt-search", 
    action="store_true", 
    help="Whether to use yt-dlp for searching instead of ddgs"
)

parser.add_argument(
    "--limit", "-l",
    type=int,
    default=10,
    help="Maximum number of search results to return"
)

parser.add_argument(
    "--output-folder", "-o",
    type=str,
    help="Folder to save downloaded clips and segments"
)

parser.add_argument(
    "--max-duration",
    type=str,
    default="5:00",
    help="Maximum duration of the voice download in minutes:seconds format (e.g., 5:20 for 5 minutes and 20 seconds)"
)

parser.add_argument(
    "--sample-voice",
    type=str,
    required=False,
    help="A short sample increases the quality by a lot."
)

parser.add_argument(
    "--skip-finalcheck",
    action="store_true",
    help="Whether to skip the final check for multiple voices"
)


def main():
    args = parser.parse_args()
    WORK_DIR = args.output_folder if args.output_folder is not None else resolve_path()
    CHARACTER_NAME = args.character.replace(' ', '_')
    os.makedirs(WORK_DIR, exist_ok=True)

    if not ensure_sample_voice(args=args):
        exit(1)
    
    search_results = []
    if args.url is None:
        search_results = search_voice_samples(args.character, args.query, limit=args.limit, is_yt_search=args.use_yt_search, max_duration=args.max_duration)
    else:
        search_results = [get_yt_video_info(args.url, args.character)]
    if len(search_results) == 0:
        print("[red]Error: couldn't find any videos[/red]")
        print("[yellow]Try using different search terms or try increasing the limits or duration[/yellow]")
        exit(1)
    print(f"Found {len(search_results)} results for character '{args.character}'")
    console.print_search_results(search_results)

    print(f"Downloading {len(search_results)} clips for '{args.character}'")
    clips: list[Clip] = []
    for _, search_result in track(enumerate(search_results), description="Downloading audios..."):
        audio_file = download_audio(search_result, WORK_DIR)
        if audio_file is not None:
            clips.append(audio_file)
    print(f"Saved {len(clips)}/{len(search_results)} clips in {os.path.dirname(clips[0].path)}")

    # Join the clips and then run deizarization and segmentation on the joined clip

    print("[green]Running speaker diarization and segmentation[/green]")
    segmentation_data_list: list[SegmentData] = []
    for i, clip in enumerate(clips):
        print(f"[cyan]{i+1}/{len(clips)}[/cyan] ({clip.id}) ({clip.duration} seconds)")
        segement_data = run_diarization(clip)
        segmentation_data_list.append(segement_data)
        segment_folder = os.path.join(WORK_DIR, 'segments')

        os.makedirs(segment_folder, exist_ok=True)
        segments = run_segmentation(segement_data, segment_folder)
        join_segmented_clips(segement_data, segment_folder, clip)

        # Clean up! =============
        for segment in segments:
            os.remove(segment)
        os.rmdir(segment_folder)

    joined_path = os.path.join(WORK_DIR, f"{CHARACTER_NAME}_joined.wav")

    paths_to_keep = []
    if args.sample_voice is not None:
        paths_to_keep = find_speaker_character_with_sample(segmentation_data_list, args.sample_voice, WORK_DIR)
    else:
        paths_to_keep = find_speaker_character_3(segmentation_data_list, WORK_DIR)

    join_segments(paths_to_keep, joined_path)
    print(f"[green]Joined segments saved to {joined_path}[/green]")

    # clear up! =============
    for clip in clips:
        if os.path.exists(clip.path):
            os.remove(clip.path)
    for segment_data in segmentation_data_list:
        for i in range(segment_data.number_of_speakers):
            segment_path = os.path.join(
                os.path.dirname(segment_data.audio_path),
                f"{segment_data.clip_id}_speaker_{i}.wav"
            )
            if os.path.exists(segment_path):
                os.remove(segment_path)

    # Run diarization one last time just in case there are more than one voices.

    # Do it if sample voice isn't available or if final check is not skipped

    if args.skip_finalcheck:
        return

    joined_clip = Clip(
        id=f"{CHARACTER_NAME}_joined",
        name=args.character,
        path=joined_path,
        duration=None
    )
    print(f"[yellow]Running final check[/yellow]")
    joined_segment_data = run_diarization(joined_clip)

    final_paths: list[str] = []
    if joined_segment_data.number_of_speakers <= 1:
        final_paths.append(joined_clip.path)
    else:
        print(f"[yellow]Multiple speakers found in joined clip. Running segmentation...[/yellow]")
        joined_segment_folder = os.path.join(WORK_DIR, 'segments')
        os.makedirs(joined_segment_folder, exist_ok=True)
        joined_segments = run_segmentation(joined_segment_data, joined_segment_folder)

        save_paths = join_segmented_clips(joined_segment_data, joined_segment_folder, joined_clip)
        if len(save_paths) > 0:
            final_paths.extend(save_paths)

        # Clean up! =========================
        for segment in joined_segments:
            if os.path.exists(segment):
                os.remove(segment)
        if os.path.exists(joined_segment_folder):
            os.rmdir(joined_segment_folder)
        if os.path.exists(joined_path):
            os.remove(joined_path)

    print(f"[green]Final speaker segments: [/green]")
    print("[green]" + "\n".join([f"- `{path}`" for path in final_paths]) + "[/green]")


if __name__ == "__main__":
    main()
