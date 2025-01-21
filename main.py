import textwrap
import dspy

from dspyclasses import MultiHopRAG
from utils.llama_utils import get_vector_store_index, embed_documents, load_docs
from notlogging.notlogger import NotALogger

PG_CONNECTION_URI = "postgresql://postgres:postgres@localhost:5432"
PG_DB_NAME = "postgres"

logger = NotALogger(__name__)
logger.enable = True

def m_main():
    message_handler = {
        "query": logger.info,
        "files": logger.info,
        "image": print,
        "answer": print,
    }

    docs = load_docs('data', PG_CONNECTION_URI, PG_DB_NAME, "data_items")

    index = get_vector_store_index(docs, PG_CONNECTION_URI, PG_DB_NAME, "data_items", "ibm-granite/granite-embedding-278m-multilingual")
    multi_hop = MultiHopRAG(index=index, num_passages=5)
    chatbot = dspy.LM(
        model="ollama/llama3.1",
        system_prompt="Strictly follow the given instructions and adhere to the given format",
        base_url="http://localhost:11434/",
        cache=False,
        # model_type="chat"
    )

    dspy.settings.configure(lm=chatbot)

    _in = False
    prompt = ''
    out = None
    while True:
        if not _in:
            prompt = input(">> ").strip()
            prompt = prompt if prompt else "Hi"
            out = multi_hop.forward(prompt)
        _in = True

        try:
            msg = next(out)
            message_handler[msg['type']](msg['content'])

        except StopIteration:
            choice = input("History (Y/[N])? ").strip().lower()
            if choice == 'y':
                print(chatbot.inspect_history(n=3))
            _in = False

if __name__ == "__main__":
    m_main()
