import dspy

from dspyclasses import MultiHopRAG
from utils.llama_utils import get_vector_store_index, load_docs
from notlogging.notlogger import NotALogger

from config import config

logger = NotALogger(__name__)
logger.enabled = True

def m_main():
    message_handler = {
        "query": logger.info,
        "files": logger.info,
        "image": print,
        "answer": print,
    }

    docs = load_docs('data', config.get('PG_CONNECTION_URI'), config.get('embed_table'))

    index = get_vector_store_index(docs, config.get('PG_CONNECTION_URI'), config.get('embed_table'), config.get('embed_model'))
    multi_hop = MultiHopRAG(index=index, num_passages=5)
    chatbot = dspy.LM(
        model="ollama/"+config.get('ollama_model'),
        system_prompt="Strictly follow the given instructions and adhere to the given format",
        base_url=config.get('ollama_url'),
        cache=False,
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
    print(logger.modules)
    # m_main()
