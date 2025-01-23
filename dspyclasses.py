import dspy
import json

from llama_index.core import VectorStoreIndex
from signatures import GenerateSearchQuery, GenerateAnswer, ImageRag
from dsp.utils.utils import deduplicate
from notlogging.notlogger import NotALogger

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


class ImagePlacerRag(dspy.Module):
    def __init__(self):
        super().__init__()
        self.image_rag = dspy.ChainOfThought(ImageRag)

    def forward(self, ground_truth, question):
        resp = self.image_rag(ground_truth=ground_truth, question=question)
        return dspy.Prediction(response=resp.response)


class MultiHopRAG(dspy.Module):
    def __init__(self, index: VectorStoreIndex, num_passages=3):
        super().__init__()

        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.retrieve = LlamaIndexRMClient(k=num_passages, index=index)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.place_images = ImagePlacerRag()
        self.place_images.load("optimized_image_rag.json")

        self.message_history: List[dict[str, str]] = []

        self.mappings = ''
        with open("labels/new.json") as f:
            self.mappings = json.dumps(json.load(f))

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

        yield {"type": "answer", "content": prediction.answer, "status": "Inserting images"}

        resp = self.place_images(ground_truth=self.mappings, question=prediction.answer)

        yield {"type": "image", "content": resp.response, "status": "DONE"}

        self.update_message_history([
            {"role": "user", "content": question},
            {"role": "assistant", "content": prediction.answer},
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

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Call forward() method directly.")
