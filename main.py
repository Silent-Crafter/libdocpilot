from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def embed_documents(embed_model, debug: bool = False):
    if debug: print("Loading data...\t\t\t", end='')
    documents = SimpleDirectoryReader("data").load_data()

    # build index of documents
    if debug: print("\rCreating Index...\t\t\t", end='')
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=OllamaEmbedding(model_name=embed_model),
    )
    if debug: print("\r"+(" "*50)+"\r", end='')

    return index

def get_query_engine(index, llm_model, debug: bool = False):
    # query engine setup
    if debug: print("\rReadying query engine...\t\t\t", end='')
    query_engine = index.as_query_engine(llm=Ollama(model=llm_model))
    if debug: print("\r"+(" "*50)+"\r", end='')
    return query_engine


def main():
    index = embed_documents("nomic-embed-text", debug=True)
    query_engine = get_query_engine(index, "llama3.2", debug=True)
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


if __name__ == "__main__":
    main()

