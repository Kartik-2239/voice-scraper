
import os

from numpy import dtype, ndarray
import numpy as np
from numpy.typing import NDArray
from voice_scraper.embedding import get_embedding, cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.neighbors import NearestNeighbors
from voice_scraper.models import SegmentData
from pydantic import BaseModel
from typing import Any, cast
from rich import print
from voice_scraper.utils import resolve_path, check_path_exists


class Comman_l(BaseModel):
    path1: str
    path2: str
    similarity: float


def _find_sharp_drop_cutoff(similarities: list[float]) -> int:
    """Return how many leading items to keep before the sharpest drop.

    Given similarity scores sorted in descending order, find the largest gap
    between two consecutive values and cut there. So a sequence like
    0.80, 0.75, 0.71, 0.08 keeps the first three and stops before 0.08.
    """
    if len(similarities) <= 1:
        return len(similarities)
    largest_gap = -1.0
    cutoff = len(similarities)
    for i in range(len(similarities) - 1):
        gap = similarities[i] - similarities[i + 1]
        if gap > largest_gap:
            largest_gap = gap
            cutoff = i + 1
    return cutoff


def find_speaker_character_3(segmentation_data_list: list[SegmentData], path: str|None = None) -> list[str]:
    if path is None:
        path = resolve_path()   
    clip_id_to_paths: dict[str, list[str]] = {}
    path_to_embedding: dict[str, NDArray[Any]] = {}
    path_to_clip_id: dict[str, str] = {}
    for segmentation_data in segmentation_data_list:
        for segment in segmentation_data.segments:
            if clip_id_to_paths.get(segmentation_data.clip_id) is None:
                clip_id_to_paths[segmentation_data.clip_id] = []
            req_path = os.path.join(path, f"{segment.clip_id}_speaker_{segment.speaker_label}.wav")
            if not check_path_exists(req_path):
                print(f"[yellow]Warning: Path '{req_path}' does not exist. Skipping this segment.[/yellow]")
                continue
            if req_path not in clip_id_to_paths[segmentation_data.clip_id]:
                clip_id_to_paths[segmentation_data.clip_id] += [req_path]
                path_to_clip_id[req_path] = segmentation_data.clip_id

    for _, paths in clip_id_to_paths.items():
        for current_path in paths:
            path_to_embedding[current_path] = np.array(get_embedding(current_path)).reshape(-1)

    all_paths = list(path_to_embedding.keys())
    n = len(all_paths)
    if n < 2:
        print("[yellow]Not enough speaker files found to compare.[/yellow]")
        return []

    clip_ids = [path_to_clip_id[p] for p in all_paths]

    # embeddings shape: (n_samples, embedding_dim)
    embeddings = np.array([path_to_embedding[p] for p in all_paths])
    sim = sklearn_cosine_similarity(embeddings)
    np.fill_diagonal(sim, -1)

    # For every node keep only its strongest cross-clip neighbours, cutting at
    # the sharpest similarity drop so weak links never enter the chain.
    adjacency: dict[int, list[tuple[int, float]]] = {}
    for i in range(n):
        candidates = [
            (j, float(sim[i][j]))
            for j in range(n)
            if clip_ids[j] != clip_ids[i]
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        cutoff = _find_sharp_drop_cutoff([c[1] for c in candidates])
        adjacency[i] = candidates[:cutoff]

    best_chain: list[tuple[int, int, float]] = []
    best_score = -1.0

    def dfs(node: int, used_clips: set[str], chain: list[tuple[int, int, float]], score: float) -> None:
        nonlocal best_chain, best_score
        if (len(chain), score) > (len(best_chain), best_score):
            best_chain = list(chain)
            best_score = score
        for neighbour, edge_sim in adjacency[node]:
            neighbour_clip = clip_ids[neighbour]
            if neighbour_clip in used_clips:
                continue
            chain.append((node, neighbour, edge_sim))
            used_clips.add(neighbour_clip)
            dfs(neighbour, used_clips, chain, score + edge_sim)
            used_clips.remove(neighbour_clip)
            chain.pop()

    for start in range(n):
        dfs(start, {clip_ids[start]}, [], 0.0)

    chain_list: list[Comman_l] = [
        Comman_l(
            path1=all_paths[i],
            path2=all_paths[j],
            similarity=edge_sim,
        )
        for i, j, edge_sim in best_chain
    ]

    if chain_list:
        print(f"[green]Joining {len(chain_list) + 1} clips[/green]")
    else:
        print("[yellow]No matching speakers found across different clips.[/yellow]")

    paths2: list[str] = []
    for common in chain_list:
        if common.path1 not in paths2:
            paths2.append(common.path1)
        if common.path2 not in paths2:
            paths2.append(common.path2)
    return paths2

def find_speaker_character_with_sample(segmentation_data_list: list[SegmentData], sample_path: str, path: str | None = None) -> list[str]:
    if path is None:
        path = resolve_path()
    clip_id_to_paths: dict[str, list[str]] = {}
    path_to_embedding: dict[str, NDArray[Any]] = {}
    path_to_clip_id: dict[str, str] = {}
    for segmentation_data in segmentation_data_list:
        for segment in segmentation_data.segments:
            if clip_id_to_paths.get(segmentation_data.clip_id) is None:
                clip_id_to_paths[segmentation_data.clip_id] = []
            req_path = os.path.join(path, f"{segment.clip_id}_speaker_{segment.speaker_label}.wav")
            if not check_path_exists(req_path):
                print(f"[yellow]Warning: Path '{req_path}' does not exist. Skipping this segment.[/yellow]")
                continue
            if req_path not in clip_id_to_paths[segmentation_data.clip_id]:
                clip_id_to_paths[segmentation_data.clip_id] += [req_path]
                path_to_clip_id[req_path] = segmentation_data.clip_id

    for _, paths in clip_id_to_paths.items():
        for current_path in paths:
            path_to_embedding[current_path] = np.array(get_embedding(current_path)).reshape(-1)

    embeddings = np.array([path_to_embedding[p] for p in path_to_embedding])
    sample_path_embedding = np.array(get_embedding(sample_path)).reshape(1, -1)
    nn = NearestNeighbors(n_neighbors=len(segmentation_data_list), metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(sample_path_embedding)
    paths_to_return: list[str] = []
    for index in indices[0]:
        paths_to_return.append(list(path_to_embedding.keys())[index])
    
    return paths_to_return
