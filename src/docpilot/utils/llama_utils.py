import os
import logging
import psycopg2

from os import PathLike
from llama_index.core import VectorStoreIndex, Document, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.node_parser import SemanticSplitterNodeParser
from sqlalchemy import make_url
from docpilot.parsers import CustomXLSXReader, CustomPDFReader
from docpilot.utils.embed_utils import get_embedder

from typing import List, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def get_indexed_nodes(uri: str, embedding_table: str) -> List[str]:
    """
    Get a list of files/nodes that are already indexed in Vector Store
    :param uri: PGVectorStore URI
    :param embedding_table: table containing the embeddings
    :return: List of indexed nodes
    """
    result = []
    try:
        conn = psycopg2.connect(uri)
        cursor = conn.cursor()
        # Get names of all files that are indexed
        cursor.execute(f"SELECT DISTINCT metadata_->>'file_name' FROM {embedding_table}")
        result = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")

    return list(map(lambda x: x[0], result)) if result else []


def load_docs(
        doc_dir: Union[PathLike[str], str],
        uri: str,
        embedding_table: str,
        **kwargs
) -> List[Document]:
    """
    Load documents from a directory using SimpleDirectoryReader of LlamaIndex
    :param doc_dir: the directory to load documents from
    :param kwargs: additional arguments to pass to SimpleDirectoryReader
    :param uri: the uri of pgvector database
    :param db: the database name
    :param embedding_table: the table name of embeddings
    :return: List of Document
    """

    input_files = kwargs.pop("input_files", [])

    file_extractors = {
        ".xlsx": CustomXLSXReader(),
        ".pdf": CustomPDFReader(),
    }

    if input_files:
        files = [input_files]
    else:
        # List files only. Skip hidden files of linux system
        files = list(filter(
            lambda f: f[0] != '.' and os.path.isfile(os.path.join(doc_dir, f)),
            os.listdir(doc_dir)
        ))

    indexed_nodes = get_indexed_nodes(uri, embedding_table)
    logger.debug("Already embedded files: %s", indexed_nodes)
    logger.debug("Input files: %s", files)

    # Exclude files that are already indexed and return a path of the file
    input_files = list(map(
        lambda f: os.path.join(doc_dir, f),
        (
            filter(lambda f: f not in indexed_nodes, files)
            if indexed_nodes
            else files
        )
    ))

    if not input_files:
        return []

    return SimpleDirectoryReader(
        input_dir=doc_dir,
        file_extractor=file_extractors,
        input_files=input_files,
        **kwargs
    ).load_data()


def get_vector_store_index(
        documents: List[Document],
        embed_model: str,
        uri: Optional[str] = None,
        embeddings_table: Optional[str] = None,
        reindex: Optional[bool] = False,
        embed_dim: Optional[int] = 768,
        **kwargs
) -> VectorStoreIndex:
    """
    Build or update a VectorStoreIndex backed by PGVectorStore.
    Falls back to an in-memory index if the database is unreachable.

    :param documents: list of LlamaIndex Documents to index
    :param uri: Postgres URI for PGVectorStore
    :param embeddings_table: table storing the embeddings
    :param embed_model: HuggingFace embedding model name
    :param reindex: if True, truncate existing embeddings and rebuild from scratch
    :param embed_dim: embedding dimension (used when creating the table)
    :return: VectorStoreIndex ready for use in MultihopRAG
    """
    model_cache_folder = kwargs.pop("model_cache_folder", "models/")
    device = kwargs.pop("device", "cpu")
    show_progress = kwargs.pop("show_progress", True)

    embed_model_instance = get_embedder(embed_model, device=device, model_cache_folder=model_cache_folder)[0]

    if not (uri or embeddings_table):
        return embed_documents(documents, embed_model_instance, show_progress=show_progress)

    try:
        if reindex:
            # Purge existing embeddings and rebuild from scratch
            conn = psycopg2.connect(uri)
            cursor = conn.cursor()
            try:
                cursor.execute("TRUNCATE TABLE {}".format(embeddings_table))
                conn.commit()
            except psycopg2.errors.UndefinedTable:
                logger.error(f"Undefined table '{embeddings_table}'")
            finally:
                cursor.close()
                conn.close()

            storage_context, _ = get_vector_storage_context(
                uri, embeddings_table, embed_dim=embed_dim, perform_setup=True
            )
            logger.info(f"Embedding {len(documents)} documents...")
            return embed_documents(documents, embed_model_instance, storage_context=storage_context, show_progress=show_progress)

        # Incremental indexing: only embed files not yet in the store
        indexed_nodes = get_indexed_nodes(uri, embeddings_table)
        files_to_index = [
            doc for doc in documents
            if doc.metadata.get("file_name") not in indexed_nodes
        ]

        storage_context, vector_store = get_vector_storage_context(
            uri, embeddings_table, perform_setup=False
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model_instance,
            storage_context=storage_context,
            show_progress=show_progress,
        )

        if files_to_index:
            logger.debug("Indexing %d new file(s)...", len(files_to_index))
            for doc in files_to_index:
                index.insert(doc)

        return index

    except Exception as e:
        logger.error(f"PGVectorStore unavailable ({e}), falling back to in-memory index.")
        return embed_documents(documents, embed_model_instance, show_progress=show_progress)


def embed_documents(
        documents: List[Document],
        embed_model: HuggingFaceEmbedding,
        storage_context: Optional[StorageContext] = None,
        **kwargs
) -> VectorStoreIndex:
    """
    Generate embeddings using Semantic Chunking and store in the vector store. Will store the embeddings regardless of whether they are already present.
    :param documents: list of documents
    :param embed_model:  HuggingFace embedding model
    :param storage_context:  the StorageContext instance to be used to store the embeddings in a vector store
    :param kwargs: Additional arguments/options to be passed to VectorStoreIndex.from_documents()
    :return: the index
    """
    show_progress = kwargs.pop("show_progress", True)


    splitter=SemanticSplitterNodeParser(
        buffer_size=3,
        breakpoint_percentile_threshold=90,
        embed_model=embed_model_instance
    )

    logger.info(f"Chunking {len(documents)} documents...")
    nodes=splitter.get_nodes_from_documents(documents)
    logger.info(f"Generated {len(nodes)} semantic nodes.")

    return VectorStoreIndex.from_documents(
        nodes,
        embed_model=embed_model,
        show_progress=show_progress,
        storage_context=storage_context,
        **kwargs,
    )
