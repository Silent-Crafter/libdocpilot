import dspy

from llama_index.core import VectorStoreIndex
from signatures import GenerateSearchQuery, GenerateAnswer
from dsp.utils.utils import deduplicate

from typing import Union, Optional, List


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
            lambda node: node.score >= 0.6,
            nodes
        ))

        passages = list(map(
            lambda node: node.text,
            good_nodes
        ))

        files = list(map(
            lambda node: node.metadata["file_name"],
            good_nodes
        ))

        files = deduplicate(files)

        return dspy.Prediction(
            passages=list(reversed(passages)),
            files=files,
        )


class MultiHopRAG(dspy.Module):
    def __init__(self, index: VectorStoreIndex, num_passages=3, max_hops=3):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = LlamaIndexRMClient(k=num_passages, index=index)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer, temperature=0.4)
        self.max_hops = max_hops

        self.message_history: List[dict[str, str]] = []

        self.context = []
        self.files = []

    def forward(self, question):
        context = []
        files = []

        for hop in range(self.max_hops):
            query_resp = self.generate_query[hop](context=context, question=question)
            query = query_resp.keywords
            # print(f"Searching with {query=}")
            nodes = self.retrieve(query)
            passages = nodes.passages
            files = deduplicate(files + nodes.files)
            context = deduplicate(context + passages)

        self.context = deduplicate(self.context + context)
        self.files = deduplicate(self.files + files)

        prediction = self.generate_answer(context=self.context, question=question, messages=self.format_history())

        self.update_message_history([
            {"role": "user", "content": question},
            {"role": "assistant", "content": prediction.answer},
        ])

        return dspy.Prediction(context=context, answer=prediction.answer, sources=self.files)

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
