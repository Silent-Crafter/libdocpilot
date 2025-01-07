import textwrap
import dspy

from llama_index.core import SimpleDirectoryReader

from parsers import CustomXLSXReader, CustomPDFReader
from dspyclasses import LlamaIndexRMClient, MultiHopRAG
from utils import reindex_vector_store, create_vector_store

import pathlib

PG_CONNECTION_URI = "postgresql://postgres:postgres@localhost:5432"
PG_DB_NAME = "postgres"

def m_main():
    docs = SimpleDirectoryReader("data", file_extractor={".xlsx": CustomXLSXReader()}).load_data()

    index = reindex_vector_store(docs, PG_CONNECTION_URI, PG_DB_NAME, "data_items", "ibm-granite/granite-embedding-278m-multilingual")

    multi_hop = MultiHopRAG(index=index, num_passages=10, max_hops=1)

    chatbot = dspy.LM(
        model="ollama/granite-3.1-8b-instruct",
        system_prompt="Strictly follow the given instructions and adhere to the given format",
        base_url="http://192.168.0.124:11434/",
    )

    dspy.settings.configure(lm=chatbot, rm=LlamaIndexRMClient)

    while True:
        out = multi_hop(input(">> ").strip())
        print(out.answer)
        choice = input("History ([Y]/N)? ").strip().lower()
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

    document = CustomPDFReader().load_data("data/" + files[choice-1])

    index = reindex_vector_store(document, PG_CONNECTION_URI, PG_DB_NAME, "data_items", "ibm-granite/granite-embedding-278m-multilingual")

    multi_hop = MultiHopRAG(index=index, num_passages=10, max_hops=1)

    chatbot = dspy.LM(
        model="ollama/llama3.1",
        system_prompt="Strictly follow the given instructions and adhere to the given format",
        base_url="http://192.168.0.124:11434/",
    )

    dspy.settings.configure(lm=chatbot, rm=MultiHopRAG)

    while True:
        out = multi_hop(input(">> ").strip())
        print(out.answer)
        choice = input("History ([Y]/N)? ").strip().lower()
        if choice == 'n': continue
        print(chatbot.inspect_history())
        print()


if __name__ == "__main__":
    m_main()
