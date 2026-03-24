import torch

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import Config

from typing import List, Callable, Optional, Tuple

from config import Config

def get_embedder(model_name: Optional[str] = None, **kwargs) -> Tuple[HuggingFaceEmbedding, Callable[[str], torch.Tensor]]:
    """
    Returns an embedder provider of llama_index and a callable that takes a string as an argument to embed the string.
    :param model_name: name of the embed model
    :param kwargs: additional kwargs for ollama or huggingface embedding
    :return:
    """
    if model_name is None:
        model_name = Config.embed_model
    
    trust_remote_code = kwargs.pop("trust_remote_code", True)
    cache_folder = kwargs.pop("cache_folder", "models/")
    device = kwargs.pop('device', 'cpu')

    embedder = HuggingFaceEmbedding(
        model_name=model_name, 
        trust_remote_code=trust_remote_code, 
        cache_folder=cache_folder,
        device=device,
        **kwargs
    )

    def embed(x1: str) -> torch.Tensor:
        return torch.asarray(embedder.get_text_embedding(x1))

    return embedder, embed


def compute_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 0,
                       eps: float = 1e-4) -> torch.Tensor:
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
