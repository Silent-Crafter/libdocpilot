from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import Config

from typing import Optional

class Embedder:
    embed_model_instance: Optional[HuggingFaceEmbedding] = None

    @classmethod
    def get_embedder(cls, model_name: Optional[str] = None, **kwargs) -> HuggingFaceEmbedding:
        """
        Returns an embedder provider of llama_index and a callable that takes a string as an argument to embed the string.
        :param model_name: name of the embed model
        :param kwargs: additional kwargs for ollama or huggingface embedding
        :return:
        """
        if model_name is None:
            model_name = Config.embed_model

        if cls.embed_model_instance is not None and cls.embed_model_instance.model_name == model_name:
            return cls.embed_model_instance

        trust_remote_code = kwargs.pop("trust_remote_code", True)
        cache_folder = kwargs.pop("model_cache_folder", "models/")
        device = kwargs.pop('device', 'cpu')

        cls.embed_model_instance = HuggingFaceEmbedding(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            cache_folder=cache_folder,
            device=device,
            **kwargs
        )

        return cls.embed_model_instance

