from datetime import datetime
from voice_scraper.utils import resolve_duration

from ddgs import DDGS
from voice_scraper.models import SearchResults
import argparse
import yt_dlp
from functools import partial
import os

MAX_TRIES = int(os.getenv("MAX_TRIES", "3"))

def generate_id(character: str) -> str:
    return f"{character.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

def search_voice_samples(
    character: str,
    search_terms: list[str],
    limit: int = 10,   
    max_duration: str | None = "5:00",
    is_yt_search: bool = False
)-> list[SearchResults]:
    
    if is_yt_search:
        results = yt_search(character, search_terms, max_duration, limit)
        for i in range(1, MAX_TRIES+1):
            if len(results) < limit:
                results = yt_search(character, search_terms, max_duration, limit + i*(limit - len(results)))
            else:
                break
        return results[:limit]
    else:
        results =  ddg_search(character, search_terms, max_duration,  limit)
        for i in range(1, MAX_TRIES+1):
            if len(results) < limit:
                results = ddg_search(character, search_terms, max_duration, limit + i*(limit - len(results)))
            else:
                break
        return results[:limit]

def ddg_search(
    character: str,
    search_terms: list[str],
    max_duration: str | None = "5:00",
    limit: int = 10,
)-> list[SearchResults]:
    ddgs = DDGS()
    results = ddgs.videos(f"{character} {' '.join(search_terms)}", max_results=limit)

    search_results: list[SearchResults] = []
    for result in results:
        duration = result.get("duration") # 12:12
        total_seconds = resolve_duration(duration) if duration is not None else None
        max_seconds = resolve_duration(max_duration) if max_duration is not None else None
        if max_seconds is not None and total_seconds is not None and total_seconds > max_seconds:
            continue
        search_results.append(SearchResults(
            id=generate_id(character),
            duration=total_seconds if duration is not None else None,
            url=result.get("content") or "",
            name=character,
            content=result.get("content") or "",
        ))
    search_results.sort(key=lambda x: x.duration or 0)
    return search_results


def yt_search(character: str, search_terms: list[str], max_duration: str | None = None, limit: int = 10) -> list[SearchResults]:
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "match_filter": yt_dlp.utils.match_filter_func(f'duration <= {max_duration}')} ) as ydl: # type: ignore
        info = ydl.extract_info(f"ytsearch{limit}:{character} {' '.join(search_terms)}", download=False)
    search_results: list[SearchResults] = []
    for entry in info.get("entries", []):
        search_results.append(SearchResults(
            id=generate_id(character),
            duration=entry.get("duration") or None,
            url=entry.get("webpage_url") or "",
            name=character,
            content=entry.get("title") or "",
        ))
    search_results.sort(key=lambda x: x.duration or 0)    
    return search_results

args = argparse.ArgumentParser()
args.add_argument("--character", type=str, required=True, help="The character to search for")
args.add_argument("--search-terms", type=str, nargs="+", default=["talking", "speaking", "podcast"], help="Additional search terms to include in the search query")
args.add_argument("--limit", type=int, default=10, help="The maximum number of search results to return")
args.add_argument("--use-yt-search", action="store_true", help="Whether to use yt-dlp for searching instead of ddgs")

if __name__ == "__main__":
    args = args.parse_args()

    if args.character is None:
        print("Please provide a character to search for using the --character flag")
        exit(1)

    if args.use_yt_search:
        print("\n-------------\n".join([str(result) for result in yt_search(args.character, args.search_terms, limit=args.limit)]))
    else:
        print("\n-------------\n".join([str(result) for result in ddg_search(args.character, args.search_terms, limit=args.limit)]))
