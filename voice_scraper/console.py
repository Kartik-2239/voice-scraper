from rich.table import Table
from rich.console import Console
from voice_scraper.models import SearchResults


def print_search_results(search_results: list[SearchResults]) -> None:
    table = Table()
    table.add_column("ID", justify="center", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Duration", style="cyan")
    table.add_column("URL", style="blue", no_wrap=True)

    for i, result in enumerate(search_results):
        duration_str = f"{result.duration // 60}:{result.duration % 60:02}" if result.duration is not None else "N/A"
        table.add_row(str(i+1), result.name, duration_str, result.url)

    console = Console()
    console.print(table)