import os
import psycopg2


from os import PathLike
from llama_index.core import VectorStoreIndex, Document, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.node_parser import SemanticSplitterNodeParser
from sqlalchemy import make_url
from docpilot.parsers import CustomXLSXReader, CustomPDFReader
from docpilot.notlogging.notlogger import NotALogger

from typing import List, Optional, Union, Tuple

logger = NotALogger(__name__)
logger.enabled = False


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
    logger.log(f"Already embedded files: {indexed_nodes}", "debug")
    logger.log(f"Input files: {files}", "debug")

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


def get_index_from_store(
        vector_store: BasePydanticVectorStore,
        storage_context: StorageContext,
        embed_model: str,
        **kwargs
) -> VectorStoreIndex:
    model_cache_folder = kwargs.get("model_cache_folder", "models/")

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=HuggingFaceEmbedding(
            model_name=embed_model,
            cache_folder=model_cache_folder,
            trust_remote_code=True,
        ),
        storage_context=storage_context,
        show_progress=True,
        **kwargs
    )


def get_vector_store_index(
        documents: List[Document],
        uri: str,
        embeddings_table: str,
        embed_model: str,
        reindex: Optional[bool] = False,
) -> VectorStoreIndex:

    # Reindex the vector store i.e. generate all embeddings again
    if reindex:
        return reindex_vector_store(documents, uri, embeddings_table, embed_model)

    nodes = get_indexed_nodes(uri, embeddings_table)

    # Index/Embed files that haven't already been indexed
    files_to_index = list(filter(lambda node: node.metadata['file_name'] not in nodes, documents))

    storage_context, vs = get_vector_storage_context(uri, embeddings_table, perform_setup=False)
    index = get_index_from_store(vs, storage_context, embed_model)

    # If new files are found, generate embeddings and store in index
    if files_to_index:
        logger.log(f"Found following extra files to index: {files_to_index}", "debug")
        for file in files_to_index:
            index.insert(file)

    return index


def reindex_vector_store(
        documents: List[Document],
        uri: str,
        embeddings_table: str,
        embed_model: str,
        embed_dim: Optional[int] = 768
) -> VectorStoreIndex:
    conn = psycopg2.connect(uri)
    cursor = conn.cursor()

    # Purge all existing embeddings
    try:
        cursor.execute('TRUNCATE TABLE {}'.format(embeddings_table))
        conn.commit()
    except psycopg2.errors.UndefinedTable:
        logger.error(f"Undefined table '{embeddings_table}'")

    cursor.close()
    conn.close()

    storage_context, _ = get_vector_storage_context(uri, embeddings_table, embed_dim=embed_dim, perform_setup=True)

    logger.info(f"Embedding {len(documents)} documents...")
    return embed_documents(documents, embed_model, storage_context)


def embed_documents(
        documents: List[Document],
        embed_model: str,
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
    model_cache_folder = kwargs.get("model_cache_folder", "../models/")
    show_progress = kwargs.get("show_progress", True)
    device = kwargs.get("device", "cpu")

    logger.info(f"Initializing Embedding Model: {embed_model}")

    embed_model_instance=HuggingFaceEmbedding(
        model_name=embed_model,
        cache_folder=model_cache_folder,
        trust_remote_code=True,
        device=device,
    )

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
        embed_model=embed_model_instance,
        show_progress=show_progress,
        storage_context=storage_context,
        **kwargs,
    )


def get_vector_storage_context(uri: str, table: str, **kwargs) -> Tuple[StorageContext, PGVectorStore]:
    """
    Create a vector store to store embeddings from a URI and a database and table
    :param uri: Postgres URI
    :param db: Postgres database
    :param table: Postgres table
    :param kwargs: extra configuration options like embed_dim, etc.
    :return: The vector store and storage context
    """
    embed_dim = kwargs.pop('embed_dim', 768)
    hnsw_kwargs = kwargs.pop('hnsw_kwargs', {
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    })

    url = make_url(uri)
    vector_store = PGVectorStore.from_params(
        database=url.database,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=table.replace("data_", ""),
        embed_dim=embed_dim,
        hnsw_kwargs=hnsw_kwargs,
        **kwargs
    )

    return StorageContext.from_defaults(vector_store=vector_store), vector_store
