import os

from pyannote.audio import Model
from pyannote.audio import Inference
from dotenv import load_dotenv
load_dotenv()

def load_model() -> Model:
    model_name = os.getenv("EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM")
    model = Model.from_pretrained(model_name)
    if model is None:
        print("Failed to load the model. Please check your Huggingface token and internet connection.")
        exit(1)
    return model

def get_embedding(audio_path: str):
    model = load_model()
    inference = Inference(model, window="whole")
    embedding = inference(audio_path)
    return embedding


def cosine_similarity(embedding1, embedding2) -> float:
    from scipy.spatial.distance import cdist
    similarity = 1 - cdist(embedding1.reshape(1, -1), embedding2.reshape(1, -1), metric="cosine")[0,0] # type: ignore
    return similarity
