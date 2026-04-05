import json
import logging
import dspy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from docpilot.dspyclasses import MultiHopRAG
from docpilot.utils.llama_utils import get_vector_store_index, load_docs
from docpilot.utils.logger import setup_logging

from config import Config as config

logger = logging.getLogger(__name__)

def to_html_file(data: str):
    with open("example.html", "w", encoding="utf-8") as f:
        data = """ <html>
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
        "query": lambda msg: logger.info("Query: %s", msg),
        "files": lambda msg: logger.info("Files: %s", msg),
        "answer_with_images": to_html_file,
        "answer": print,
    }

    docs, image_docs, image_mappings = load_docs('data', config.PG_CONNECTION_URI, config.embed_table)

    with open("./labels/image_mappings_debug.json", "w") as f:
        json.dump(image_mappings, f, indent=2)

    logger.info("Dumped %d image mappings to labels/image_mappings_debug.json", len(image_mappings))

    index = get_vector_store_index(docs, config.embed_model, embeddings_table=config.embed_table, uri=config.PG_CONNECTION_URI)
    image_index = get_vector_store_index(image_docs, config.embed_model, embeddings_table="data_images", uri=config.PG_CONNECTION_URI)

    multi_hop = MultiHopRAG(index=index, image_index=image_index, num_passages=3)
    chatbot = dspy.LM(
        model="ollama/"+config.ollama_model,
        system_prompt="Strictly follow the given instructions and adhere to the given format",
        base_url=config.ollama_url,
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
    setup_logging()
    m_main()

