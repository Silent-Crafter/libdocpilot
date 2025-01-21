import os

import psycopg2

from llama_index.core import VectorStoreIndex, Document, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import make_url
from parsers import CustomXLSXReader, CustomPDFReader
from notlogging.notlogger import NotALogger

from typing import List, Optional, Union

logger = NotALogger(__name__)
logger.enable = False


def get_indexed_nodes(url: str, db: str, embedding_table: str) -> Union[List[str], None]:
    result = []
    try:
        conn = psycopg2.connect(url + '/' + db)
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT metadata_->>'file_name' FROM {embedding_table}")
        result = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error("Unexpected Error: ", e)

    if result:
        return list(map(lambda x: x[0], result))


def load_docs(doc_dir: str, uri: str, db: str, embedding_table: str, **kwargs) -> List[Document]:
    """
    Load documents from a directory using SimpleDirectoryReader of LlamaIndex
    :param doc_dir: the directory to load documents from
    :param kwargs: additional arguments to pass to SimpleDirectoryReader
    :param uri: the uri of pgvector database
    :param db: the database name
    :param embedding_table: the table name of embeddings
    :return: List of Document
    """
    file_extractors = {
        ".xlsx": CustomXLSXReader(),
        ".pdf": CustomPDFReader(),
    }

    # Smart loading
    files = filter(lambda f: f[0] != '.' and os.path.isfile(os.path.join(doc_dir, f)), os.listdir(doc_dir))
    indexed_nodes = get_indexed_nodes(uri, db, embedding_table)

    if indexed_nodes:
        input_files = filter(lambda f: f not in indexed_nodes, files)
    else:
        input_files = files

    input_files = list(map(lambda f: os.path.join(doc_dir, f), input_files))

    return SimpleDirectoryReader(
        doc_dir,
        file_extractor=file_extractors,
        input_files=input_files,
        **kwargs
    ).load_data()


def get_total_nodes(conn, table, col="id"):
    cursor = conn.cursor()
    cursor.execute(f"SELECT count({col}) FROM {table}")
    return cursor.fetchone()[0]


def get_index_from_store(vector_store, storage_context, embed_model, **kwargs) -> VectorStoreIndex:
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
        db: str,
        embeddings_table: str,
        embed_model: str,
        reindex: Optional[bool] = False,
) -> VectorStoreIndex:

    nodes = get_indexed_nodes(uri, db, embeddings_table)

    # Index files that haven't already been indexed
    # if nodes:
    files_to_index = list(filter(lambda node: node.metadata['file_name'] not in nodes if nodes else True, documents))
    # else:
    #     files_to_index = documents

    # Otherwise retrieve all the documents
    if reindex:
        return reindex_vector_store(documents, uri, db, embeddings_table, embed_model)

    storage_context, vc = get_vector_storage_context(uri, db, embeddings_table, perform_setup=False)
    index = get_index_from_store(vc, storage_context, embed_model)

    if files_to_index:
        logger.log(f"Found following extra files to index: {files_to_index}", "debug")
        for file in files_to_index:
            index.insert(file)

    return index


def reindex_vector_store(documents: List[Document], uri, db, embeddings_table: str, embed_model: str,
                         embed_dim: Optional[int] = 768) -> VectorStoreIndex:
    conn = psycopg2.connect(uri)
    cursor = conn.cursor()

    try:
        cursor.execute('TRUNCATE TABLE {}'.format(embeddings_table))
        conn.commit()
    except psycopg2.errors.UndefinedTable:
        pass

    cursor.close()
    conn.close()

    storage_context, _ = get_vector_storage_context(uri, db, embeddings_table, embed_dim=embed_dim, perform_setup=False)

    return embed_documents(documents, embed_model, storage_context)


def embed_documents(documents: List[Document], embed_model: str, storage_context: StorageContext,
                    **kwargs) -> VectorStoreIndex:
    """
    Generate embeddings and store in the vector store. Will store the embeddings regardless of whether they are already present.
    :param documents: list of documents
    :param embed_model:  HuggingFace embedding model
    :param storage_context:  the StorageContext instance to be used to store the embeddings in a vector store
    :param kwargs: Additional arguments/options to be passed to VectorStoreIndex.from_documents()
    :return: the index
    """
    model_cache_folder = kwargs.get("model_cache_folder", "../models/")
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


def get_vector_storage_context(uri, db, table, **kwargs) -> (StorageContext, PGVectorStore):
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
        database=db,
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
