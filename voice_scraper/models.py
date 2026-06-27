from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SearchResults(BaseModel):
    id: str
    url: str
    duration: int | None
    name: str
    content: str | None

class Clip(BaseModel):
    id: str
    name: str
    path: str
    duration: int | None

class Segment(BaseModel):
    id: str
    clip_id: str
    start_time: float
    end_time: float
    speaker_label: int

class SegmentData(BaseModel):
    id: str
    clip_id: str
    number_of_speakers: int
    audio_path: str
    segments: list[Segment]