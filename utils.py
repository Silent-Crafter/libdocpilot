import psycopg2

from llama_index.core import VectorStoreIndex, Document, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from parsers import CustomXLSXReader, CustomPDFReader
from sqlalchemy import make_url
from typing import List, Optional


def load_docs(dir: str, **kwargs) -> List[Document]:
    """
    Load documents from a directory using SimpleDirectoryReader of LlamaIndex
    :param dir: the directory to load documents from
    :param kwargs: additional arguments to pass to SimpleDirectoryReader
    :return: List of Document
    """
    file_extractors = {
        ".xlsx": CustomXLSXReader(),
        ".pdf": CustomPDFReader(),
    }

    return SimpleDirectoryReader(
        dir,
        file_extractor=file_extractors,
        **kwargs
    ).load_data()


def init_pg(uri):
    conn = psycopg2.connect(uri)
    conn.autocommit = True

    return conn

def get_total_nodes(conn, table, col = "id"):
    cursor = conn.cursor()
    cursor.execute(f"SELECT count({col}) FROM {table}")
    return cursor.fetchone()[0]


def get_index_from_store(vector_store, embed_model, **kwargs) -> VectorStoreIndex:
    model_cache_folder = kwargs.get("model_cache_folder", "./models/")
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=HuggingFaceEmbedding(
            model_name=embed_model,
            cache_folder=model_cache_folder,
            trust_remote_code=True,
        ),
        show_progress=True,
        **kwargs
    )


def reindex_vector_store(documents: List[Document], uri, db, embeddings_table: str, embed_model: str, embed_dim: Optional[int] = 768) -> VectorStoreIndex:
    with psycopg2.connect(uri) as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute('TRUNCATE TABLE {}'.format(embeddings_table))
                conn.commit()
            except psycopg2.errors.UndefinedTable:
                pass

    storage_context, _ = create_vector_store(uri, db, embeddings_table, embed_dim=embed_dim)

    return embed_documents(documents, embed_model, storage_context)


def embed_documents(documents: List[Document], embed_model: str, storage_context: StorageContext, **kwargs) -> VectorStoreIndex:
    """
    Generate embeddings and store in the vector store. Will store the embeddings regardless of whether they are already present.
    :param documents: list of documents
    :param embed_model:  HuggingFace embedding model
    :param storage_context:  the StorageContext instance to be used to store the embeddings in a vector store
    :param kwargs: Additional arguments/options to be passed to VectorStoreIndex.from_documents()
    :return: the index
    """
    model_cache_folder = kwargs.get("model_cache_folder", "./models/")
    show_progress = kwargs.get("show_progress", True)

    return VectorStoreIndex.from_documents(
        documents,
        embed_model=HuggingFaceEmbedding(
            model_name=embed_model,
            cache_folder=model_cache_folder,
            trust_remote_code=True,
        ),
        show_progress=show_progress,
        storage_context=storage_context,
        **kwargs,
    )

def create_vector_store(uri, db, table, **kwargs) -> (StorageContext, PGVectorStore):
    """
    Create a vector store to store embeddings from a URI and a database and table
    :param uri: Postgres URI
    :param db: Postgres database
    :param table: Postgres table
    :param kwargs: extra configuration options like embed_dim, etc
    :return: The vector store and storage context
    """
    embed_dim = kwargs.get('embed_dim', 768)
    hnsw_kwargs = kwargs.get('hnsw_kwargs', {
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    })

    url = make_url(uri)
    vector_store = PGVectorStore.from_params(
        database=db,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=table.replace("data_", ""),
        embed_dim=embed_dim,
        hnsw_kwargs=hnsw_kwargs,
    )

    return StorageContext.from_defaults(vector_store=vector_store), vector_store
