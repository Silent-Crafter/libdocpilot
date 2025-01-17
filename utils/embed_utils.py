import torch

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from typing import List, Callable, Optional

def get_embedder(model_name: str, **kwargs) -> (BaseEmbedding, Callable[[str], torch.Tensor]):
    """
    Returns a embedder provider of llama_index and a callable that takes a string as an argument to embed the string.
    :param model_name: name of the embed model
    :param kwargs: ollama_config or hf_config for Ollama and HuggingFace and additional kwargs
    :return:
    """
    provider_map = {
        "ollama": OllamaEmbedding,
        "hf": HuggingFaceEmbedding,
    }

    provider, *model = model_name.split("/")
    model = "/".join(model)

    config = {}
    if provider == "ollama":
        config["base_url"] = kwargs.pop("base_url", "http://192.168.0.124:11434")
    elif provider == "huggingface":
        config["trust_remote_code"] = kwargs.pop("trust_remote_code", True)
        config["cache_folder"] = kwargs.pop("cache_folder", "models/")
    else:
        config = {}

    embedder: BaseEmbedding = provider_map[provider](model_name=model, **config, **kwargs)

    def embed(x1: str) -> torch.Tensor:
        return torch.asarray(embedder.get_text_embedding(x1))

    return embedder, embed

def compute_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: Optional[int] = 0, eps: Optional[float] = 1e-4) -> torch.Tensor:
    return torch.nn.CosineSimilarity(dim=dim, eps=eps).forward(x1, x2)


def compute_similarity_matrix(x: int, y: int, embeddings: List[torch.Tensor]) -> torch.Tensor:
    matrix = torch.zeros(x, y)
    for i, e1 in enumerate(embeddings):
        for j, e2 in enumerate(embeddings):
            if i == j:
                ans = torch.tensor(1.0, dtype=torch.float)
            else:
                ans = compute_similarity(e1, e2) if not matrix[j, i] else matrix[j, i]
            matrix[i, j] = ans

    return matrix
