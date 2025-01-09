import textwrap
import dspy

from llama_index.core import SimpleDirectoryReader

from parsers import CustomXLSXReader, CustomPDFReader
from dspyclasses import LlamaIndexRMClient, MultiHopRAG
from utils import reindex_vector_store, get_vector_store_index

PG_CONNECTION_URI = "postgresql://postgres:postgres@localhost:5432"
PG_DB_NAME = "postgres"

def m_main():
    docs = SimpleDirectoryReader(
        "data",
        file_extractor={
            ".xlsx": CustomXLSXReader(),
            ".pdf": CustomPDFReader(),
        }
    ).load_data()

    print(len(docs))
    index = get_vector_store_index(docs, PG_CONNECTION_URI, PG_DB_NAME, "data_items", "ibm-granite/granite-embedding-278m-multilingual")
    multi_hop = MultiHopRAG(index=index, num_passages=5, max_hops=1)
    chatbot = dspy.LM(
        model="ollama/dolphin3",
        system_prompt="Strictly follow the given instructions and adhere to the given format",
        base_url="http://192.168.0.124:11434/",
        # model_type="chat"
    )

    dspy.settings.configure(lm=chatbot)

    while True:
        out = multi_hop(input(">> ").strip())
        print(out.answer)
        print("\n", "files accessed: ", out.sources, sep='')
        choice = input("History (Y/[N])? ").strip().lower()
        if choice == 'n': continue
        print(chatbot.inspect_history())
        print()

def s_main():
    import os
    files = os.listdir("data")

    for index, file in enumerate(files):
        print(f"{index+1}. {file}")

    print()
    choice = int(input("Choose a file: ").strip())

    # document = CustomPDFReader().load_data("data/" + files[choice-1])
    document = SimpleDirectoryReader(
        input_files=["data/"+files[choice-1]],
        file_extractor={
            ".xlsx": CustomXLSXReader(),
            ".pdf": CustomPDFReader(),
        },
    ).load_data()

    index = reindex_vector_store(document, PG_CONNECTION_URI, PG_DB_NAME, "data_items", "ibm-granite/granite-embedding-278m-multilingual")
    multi_hop = MultiHopRAG(index=index, num_passages=10, max_hops=1)
    chatbot = dspy.LM(
        model="ollama/dolphin3",
        system_prompt="Strictly follow the given instructions and adhere to the given format",
        base_url="http://192.168.0.124:11434/",
        model_type="chat"
        # base_url="http://localhost:11435/",
    )
    dspy.settings.configure(lm=chatbot)

    while True:
        out = multi_hop(input(">> ").strip())
        print(out.answer)
        print("\n", "files accessed: ", out.sources)
        choice = input("History (Y/[N])? ").strip().lower()
        if choice == 'n': continue
        print(chatbot.inspect_history())
        print()
        print(*chatbot.history)


if __name__ == "__main__":
    m_main()
