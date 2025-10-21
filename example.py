import json
import dspy

from docpilot.dspyclasses import MultiHopRAG
from docpilot.utils.llama_utils import get_vector_store_index, load_docs
from docpilot.notlogging.notlogger import NotALogger
from docpilot.utils.image_utils import mappings_to_llamaindex_document

from config import config

logger = NotALogger(__name__)
logger.enabled = True

def to_html_file(data: str):
    with open("example.html", "w", encoding="utf-8") as f:
        data = """
        <html>
        <head>
        <style>
        img { display: block; width: 200px; }
        </style>
        </head>
        <body>
        """ + data + """
        </body>
        </html>
        """
        f.write(data)

def m_main():
    message_handler = {
        "query": logger.info,
        "files": logger.info,
        "answer_with_images": to_html_file,
        "answer": print,
    }

    docs = load_docs('data', config.get('PG_CONNECTION_URI'), config.get('embed_table'))
    mapping = {}

    with open("./labels/new.json") as f:
        mapping = json.load(f)

    images = mappings_to_llamaindex_document(mapping, 'out_images')

    index = get_vector_store_index(docs, config.get('PG_CONNECTION_URI'), config.get('embed_table'), config.get('embed_model'))
    image_index = get_vector_store_index(images, config.get('PG_CONNECTION_URI'), config.get('image_table', "data_images"), config.get('embed_model'))

    multi_hop = MultiHopRAG(index=index, image_index=image_index, num_passages=5)
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
    for module in logger.modules.keys():
        logger.modules[module].enabled = True

    m_main()
