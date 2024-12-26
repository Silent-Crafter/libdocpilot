from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("data").load_data()

# embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# user query model
Settings.llm = Ollama(model="mistral")

# build index of documents
index = VectorStoreIndex.from_documents(
    documents,
)

# query engine setup
query_engine = index.as_query_engine()
while True:
    try:
        prompt = input("\n>> ")
    except (EOFError, KeyboardInterrupt):
        break

    if prompt == "/bye":
        break

    response = query_engine.query(prompt.strip())
    print()
    print(response)

