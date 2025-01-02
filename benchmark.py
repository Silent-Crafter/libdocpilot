import requests
import json
import timeit
import sys
import httpcore
import httpx

from collections import defaultdict
from time import sleep

from main import get_query_engine, embed_documents


def my_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


embedding_models = [
    "nomic-embed-text:latest",
]

queries = [
    "Which computer did Paul Grahm buy first?",
    "What was the gold standard for computers in 1980"
]

# models = requests.get('http://localhost:11434/api/tags').json()['models']
# models = list(filter(
#     lambda x: x not in embedding_models,
#     map(
#         lambda x: x['name'],
#         models
#     )
# ))

models = [
    'qwen2:latest',
    'llama3.1:latest',
    'gemma2:latest',
    'mistral:latest',
    'llama3.2:latest'
]

embeds = []

my_print("Timing embedding models...")
for em in embedding_models:
    my_print(f"\rTesting {em}\t\t\t\t\t")
    # Preload the model first
    resp = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": em, "keep_alive": "5m"}
    ).text
    with open('log.txt', 'a') as f:
        f.write(f"[{em}] "+resp+"\n\n")

    start = timeit.default_timer()
    index = embed_documents(embed_model=em, debug=True)
    end = timeit.default_timer() - start

    # Unload the model
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": em, "keep_alive": 0}
    ).text
    with open('log.txt', 'a') as f:
        f.write(f"[{em}] "+resp+"\n\n")


    embeds.append((em, end, index))

my_print("\r"+(" "*80))

embeds = sorted(embeds, key=lambda x: x[1])

my_print("RESULT:")
my_print("\n".join(map(str,embeds)))

llm_perf = defaultdict(list)

my_print("\nBenchmarking LLMs with every embedding...")
for em, _, index in embeds:
    my_print(f"TESTING: {em} \t LLM: "+" ".ljust(30), end='')
    for llm in models:
        my_print("\b"*30, end='')
        my_print(f"{llm}".ljust(30), end='')
        qe = get_query_engine(index, llm, debug = False)

        # Preload the model to find actual reponse time
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": llm, "keep_alive": "5m", "stream": False}
            ).text
        except (httpcore.ReadTimeout, httpx.ReadTimeout):
            resp = '<|TIMEOUT|>'
            sleep(5)

        with open('log.txt', 'a') as f:
            f.write(f"[{llm}] "+resp+"\n\n")

        for query in queries:
            start = timeit.default_timer()
            try:
                resp = qe.query(query)
            except (httpcore.ReadTimeout, httpx.ReadTimeout):
                resp = '<|TIMEOUT|>'
                sleep(5)

            end = timeit.default_timer() - start

            llm_perf[llm].append({
                "query": query,
                "embed_model": em,
                "response": str(resp),
                "time": end
            })

    my_print("\r" + " "*80 + "\r", end='')

my_print()
my_print(llm_perf)

with open('result.json', 'a') as f:
    f.write(json.dumps(llm_perf))

