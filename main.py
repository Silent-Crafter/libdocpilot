import dspy
import psycopg2
import textwrap
from sqlalchemy import make_url

PG_CONNECTION_URI = "postgresql://postgres:postgres@localhost:5432"
PG_DB_NAME = "postgres"

def init_pg(uri):
    conn = psycopg2.connect(uri)
    conn.autocommit = True

    return conn


def get_total_nodes(conn, table, col = "id"):
    cursor = conn.cursor()
    cursor.execute(f"SELECT count({col}) FROM {table}")
    return cursor.fetchone()[0]


def create_vector_store(uri, db, table, **kwargs):
    
    embed_dim = kwargs.get('embed_dim', 768)
    hnsw_kwargs = kwargs.get('hnsw_kwargs', {
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    })

    url = make_url(PG_CONNECTION_URI)
    vector_store = PGVectorStore.from_params(
        database=db,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=table,
        embed_dim=embed_dim,
        hnsw_kwargs=hnsw_kwargs,
    )

    return vector_store, StorageContext.from_defaults(vector_store=vector_store)


def load_documents(doc_dir, **kwargs):
    return SimpleDirectoryReader(doc_dir, **kwargs).load_data()


def embed_documents(documents, embed_model, storage, debug: bool = False):
    # build index of documents if there are no entries in the store
    if debug: print("\rCreating Index...\t\t\t", end='')

    if 'granite' in embed_model:
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=HuggingFaceEmbedding(model_name=embed_model),
            cache_folder="models/",
            show_progress=True,
            storage_context=storage,
        )
    else:
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=OllamaEmbedding(model_name=embed_model),
            show_progress=True,
            storage_context=storage,
        )

    if debug: print("\r"+(" "*50)+"\r", end='')

    return index


def get_index_from_store(vector_store, embed_model):
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=OllamaEmbedding(model_name="nomic-embed-text"), show_progress=True)


def get_query_engine(index, llm_model, debug: bool = False, **kwargs):
    # query engine setup
    if debug: print("\rReadying query engine...\t\t\t", end='')

    query_engine = index.as_query_engine(llm=Ollama(model=llm_model))

    if debug: print("\r"+(" "*50)+"\r", end='')
    return query_engine

def main():
    conn = init_pg(PG_CONNECTION_URI)
    documents = load_documents("data")
    vector_store, storage = create_vector_store(PG_CONNECTION_URI, PG_DB_NAME, "items")
    if get_total_nodes(conn, "data_items") == len(documents):
        index = get_index_from_store(vector_store, "ibm-granite/granite-embedding-125m-english")
    else:
        index = embed_documents(documents, "nomic-embed-text", storage, debug=True)
    query_engine = get_query_engine(index, "qwen2", debug=True)
    while True:
        try:
            prompt = input("\n>> ")
        except (EOFError, KeyboardInterrupt):
            break

        if prompt == "/bye":
            break

        response = query_engine.query(prompt.strip())
        print()
        print("\n".join(textwrap.wrap(str(response), width=80)))


if __name__ == "__main__":
    main()

