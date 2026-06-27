# Voice Scraper

Voice scraper is a project i made for scraping audio of specific characters from youtube.
It works by searching with duck duck go or yt-dlp search and finds character clips using query and keywords provided.
It then downloads those clips and performs diarization to identify speakers and combine the audios.

## How it works

- Search — Queries DuckDuckGo or yt-dlp for videos matching the character name and search terms
- Download — Downloads audio from the search results using yt-dlp + ffmpeg
- Diarization — Runs speaker diarization on each clip to detect how many speakers are present and which segments belong to whom
- Segmentation — Splits each clip into per-speaker segments
- Speaker identification — If a sample voice is provided, matches segments to the target speaker by voice embedding similarity; otherwise picks the most frequent speaker
- Join — Merges all matched segments into a single _joined.wav file
- Final check — Re-runs diarization on the joined file; if multiple speakers are still present, segments again and returns only the target speaker's clips

## Samples

[Tony Stark](assets/tony_stark.wav)
[Violet Evergarden](assets/violet_evergarden.wav)

## Requirements

- Python >= 3.12
- uv
- ffmpeg
- yt-dlp

## Setup

```bash
uv sync
```

## Env variables

copy .env.local to .env

```bash
# Hugging face token for accessing gated models
export HUGGINGFACE_TOKEN=hf_....
export EMBEDDING_MODEL=pyannote/wespeaker-voxceleb-resnet34-LM
export DIARIZATION_MODEL=pyannote/speaker-diarization-community-1

# Use CUDA if you have an nvidia gpu
export DEVICE=CPU

# Usage Config (can be overrided by cli arguments)
export USE_YT_SEARCH=False
export OUTPUT_DIR=./downloads

# Number of reattempts if search result fails to get the specified number of search results
export MAX_TRIES=3
```


## CLI Arguments

| Argument | Defailt | Description |
|---|---|---|
| `-c`, `--character` | none | Character/speaker name (required) |
| `-q`, `--query` | none | Search query terms (default: `talking speaking podcast`) |
| `-l`, `--limit` | 10 | Max search results |
| `-o`, `--output-folder` | none | Output directory (overrides .env) |
| `--url` | none | Direct URL to a video (overrides search) |
| `--use-yt-search` | false | Use yt-dlp for searching instead of DuckDuckGo |
| `--max-duration` | 5:00 | Max video duration in `minutes:seconds` format |
| `--sample-voice` | none | Path to a short voice sample for improved speaker identification |


## Examples

```bash
# Using yt search
uv run voice-scraper -c "Frieren" -q dub english voice --use-yt-search
```

```bash
# Help
uv run voice-scraper --help
```

```bash
# Specified limit, max duration and output directory
uv run voice-scraper -c "Frieren" -q dub english voice -l 5 --max-duration 5:00
```

## Key Points (for better quality and speed)

### 1. When sample voice isn't given
In the use case where sample voice is not provided the program checks patterns and mathed among the clips to figure out the common speaker and makes a joined clip of those.
Since this is not completely reliable there is on final step of diarization to check if there are multiple speakers in the joined clip.

### 2. Using sample voice with
If sample voice is provided the tool will use it as the ground truth and compare all voices with it in order to create the final joined clip.
```bash
uv run voice-scraper -c "Frieren" -q dub english --sample-voice path/to/sample.wav
```
Make sure the sample path is a .wav file, if it isn't run the following command before
```bash
ffmpeg -i path/to/sample_with_any_ext path/to/sample.wav
```
### 3. Using `--skip-finalcheck` with sample voice
Skip finalcheck flag skips the final diarization which is used to identify multiple users in the joined clip.
This is useful when no sample voice is given as the program would have to see patterns to figure out which speaker in each clip is the required one.
```bash
# Since sample voice is used
# --skip-finalcheck flag will skip the last check because it won't improve the quality by a lot.
uv run voice-scraper -c "Frieren" -q dub english --sample-voice path/to/sample.wav
```

## Limitations

- The tool is no where near 100% accurate, it is still mostly just guessing with a small local model and struggles when the back and forth conversations are fast.
- The quality can be improved a lot my some llm that supports multi modal input. So maybe i will add that support in the future but for now it is completely local.
- The search can sometimes fail due to the combination of queries and duration.
- yt-dlp can fail for some youtube videos due to various reasons, adding cookies might fix those errors.