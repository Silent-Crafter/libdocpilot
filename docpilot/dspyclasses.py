import dspy

from llama_index.core import VectorStoreIndex
from docpilot.signatures import GenerateSearchQuery, GenerateAnswer
from dsp.utils.utils import deduplicate
from docpilot.notlogging.notlogger import NotALogger
from docpilot.utils.image_utils import image_to_b64

from typing import Union, Optional, List

logger = NotALogger(__name__)
logger.enabled = False

class LlamaIndexRMClient(dspy.Retrieve):
    def __init__(self, index: VectorStoreIndex, k: int = 3):
        super().__init__(k=k)
        self.retriever = index.as_retriever(similarity_top_k = k)

    def forward(
            self,
            query: str = None,
            k: Optional[int] = None,
            by_prob: bool = True,
            with_metadata: bool = False,
            **kwargs,
    ) -> Union[List[str], dspy.Prediction, List[dspy.Prediction]]:

        nodes = self.retriever.retrieve(query)

        good_nodes = list(filter(
            lambda node: node.score >= 0.64,
            nodes
        ))

        passages = list(map(
            lambda node: node.text,
            good_nodes
        ))

        files = list(map(
            lambda node: (node.score, node.metadata["file_name"]),
            good_nodes
        ))

        visited = set()
        files = [f for f in files if not (f in visited or visited.add(f))]

        scores = list(map(
            lambda node: node[0],
            files
        ))

        files = list(map(
            lambda node: node[1],
            files
        ))

        return dspy.Prediction(
            passages=list(reversed(passages)),
            files=files,
            scores=scores
        )


class ImageRetriever(dspy.Retrieve):
    def __init__(self, image_index: VectorStoreIndex):
        super().__init__()
        self.retriever = image_index.as_retriever()

    def forward(
            self,
            query: str = None,
            k: Optional[int] = None,
            **kwargs
    ) -> Union[str, None]:
        nodes = self.retriever.retrieve(query)

        good_nodes = list(filter(
            lambda node: node.score >= 0.69,
            nodes
        ))

        b64_images = list(map(
            lambda node: node.metadata["file_name"],
            good_nodes
        ))

        logger.info(f"Nodes: {nodes}")
        logger.info(f"Images: {b64_images}")

        if b64_images:
            return b64_images[0]

        return None


class MultiHopRAG(dspy.Module):
    def __init__(self, index: VectorStoreIndex, image_index: VectorStoreIndex, num_passages=3):
        super().__init__()

        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.retrieve = LlamaIndexRMClient(k=num_passages, index=index)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

        self.image_retriever = ImageRetriever(image_index)

        self.message_history: List[dict[str, str]] = []

        self.context = []
        self.files = []

    def forward(self, question):
        context = []
        files = []

        query_resp = self.generate_query(context=context, question=question)
        query = query_resp.keywords
        # query = question
        yield {"type": "query", "content": query, "status": "Finding files"}

        logger.info(f"Query: {query}")
        nodes = self.retrieve(query)
        passages = nodes.passages

        files = deduplicate(files + nodes.files)
        context = deduplicate(context + passages)

        self.context = deduplicate(context + self.context)
        self.files = deduplicate(files + self.files)

        yield {"type": "files", "content": self.files, "status": "Generating answer"}

        prediction = self.generate_answer(context=self.context, question=question, messages=self.format_history())
        resp = prediction.answer

        yield {"type": "answer", "content": resp, "status": "Inserting images"}

        imaged_lines = []
        for line in resp.splitlines():
            image = self.image_retriever(line)
            print("Checking:", line)
            print("Image:", image)
            if image:
                line = f"<img src=\"data:image/jpeg;base64,{image_to_b64(image)}\">\n" + line

            imaged_lines.append(line)

        final_resp = "\n".join(imaged_lines)

        yield {"type": "answer_with_images", "content": final_resp, "status": "DONE"}

        self.update_message_history([
            {"role": "user", "content": question},
            {"role": "assistant", "content": resp},
        ])

    def update_message_history(self, messages: Union[List[dict[str, str]], str]):
        if isinstance(messages, str):
            raise NotImplementedError

        self.message_history.extend(messages)

    def format_history(self) -> str:
        if not self.message_history:
            return ""

        messages = "\n".join(map(
            lambda m: m["role"].title() + ": " + m["content"],
            self.message_history
        ))

        return messages


def configure_llm(model: str, base_url: str, cache: bool, **kwargs):
    lm = dspy.LM(model="ollama/"+model, base_url=base_url, cache=cache, **kwargs)
    dspy.settings.configure(lm=lm)
    return lm
