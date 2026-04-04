import logging

import dspy
import dspy.streaming

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import NodeWithScore, TextNode
from docpilot.signatures import GenerateSearchQuery, GenerateAnswer
from dspy.dsp.utils.utils import deduplicate
from docpilot.utils.image_utils import image_to_b64
from docpilot.utils.embed_utils import Embedder

from typing import Generator, Union, Optional, List

logger = logging.getLogger(__name__)

class LlamaIndexRMClient(dspy.Retrieve):
    def __init__(self, index: VectorStoreIndex, k: int = 3):
        super().__init__(k=k)
        self.retriever = index.as_retriever(similarity_top_k = k)

    def forward(
            self,
            query: str,
            k: Optional[int] = None,
            by_prob: bool = True,
            with_metadata: bool = False,
            **kwargs,
    ) -> dspy.Prediction:

        nodes: List[NodeWithScore] = self.retriever.retrieve(query)

        good_nodes = list(filter(
            lambda node: node.score and node.score >= 0.64,
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
            passages=passages,
            files=files,
            scores=scores
        )

    def __call__(self, *args, **kwargs) -> dspy.Prediction:
        return self.forward(*args, **kwargs)


class ImageRetriever(dspy.Retrieve):
    def __init__(self, image_index: VectorStoreIndex):
        super().__init__()
        self.retriever = image_index.as_retriever()

    def forward(
            self,
            query: str,
            k: Optional[int] = None,
            **kwargs
    ) -> List[dspy.Prediction]:
        nodes = self.retriever.retrieve(query)

        good_nodes = list(filter(
            lambda node: node.score >= 0.7 if node.score else False,
            nodes
        ))

        images: List[dspy.Prediction] = list(map(
            lambda node: dspy.Prediction(image=node.metadata["file_name"], score=node.score),
            good_nodes
        ))

        logger.info(f"query: {query}")
        logger.info(f"Nodes: {good_nodes}")
        # logger.info(f"Images: {b64_images}")

        return images

    def __call__(self, *args, **kwargs) -> List[dspy.Prediction]: return self.forward(*args, **kwargs)


class MultiHopRAG(dspy.Module):
    embed_model_instance = None

    def __init__(self, index: VectorStoreIndex, image_index: VectorStoreIndex, num_passages=3):
        super().__init__()

        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.retrieve = LlamaIndexRMClient(k=num_passages, index=index)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

        self.image_retriever = ImageRetriever(image_index)

        self.message_history: List[dict[str, str]] = []

        self.context = []
        self.files = []
        if not MultiHopRAG.embed_model_instance:
            MultiHopRAG.embed_model_instance = Embedder.get_embedder()

    def forward(self, question, stream: bool = False):
        context = []
        files = []
        resp = ""
        passages = []
        nodes = None

        def serialize_message_history(history: list[dict[str, str]]):
            return '\n'.join(map(lambda d: f"{d['role']}: {d['content']}", history))

        query_resp = self.generate_query(past_context=serialize_message_history(self.message_history), question=question)
        query = query_resp.keywords

        yield {"type": "query", "content": query, "status": "Finding files"}

        logger.info("Query: %s", query)
        nodes = self.retrieve(query)
        passages = list(reversed(nodes.passages))

        files = deduplicate(files + nodes.files)
        context = deduplicate(context + passages)

        self.context = deduplicate(context + self.context)
        self.files = deduplicate(files + self.files)

        yield {"type": "files", "content": self.files, "status": "Generating answer"}

        if stream:
            streamed_generate_answer = dspy.streamify(
                self.generate_answer,
                stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
                async_streaming=False
            )

            resp_generator: Generator = streamed_generate_answer(context=self.context, question=question, messages=self.format_history())

            for chunk in resp_generator:
                if isinstance(chunk, dspy.Prediction):
                    resp = chunk.answer
                    yield {
                        "type": "answer",
                        "content": chunk.answer,
                        "status": "Inserting images"
                    }
                    break

                yield {
                    "type": "streaming_answer",
                    "content": chunk.chunk,
                    "status": "Streaming"
                }

        else:
            prediction = self.generate_answer(
                context=self.context,
                question=question,
                messages=self.format_history()
            )
            resp = prediction.answer

            yield {"type": "answer", "content": resp, "status": "Inserting images"}

        imaged_chunks, *_ = self.create_image_chunks(resp)

        final_resp, *_  = self.place_images_from_chunks(resp, imaged_chunks)
        yield {"type": "answer_with_images", "content": final_resp, "status": "DONE"}

        self.update_message_history([
            {"role": "user", "content": question},
            {"role": "assistant", "content": resp},
        ])
        yield {"type": "finalization", "content": None, "status": "DONE"}


    def create_image_chunks(self, resp):
        splitter = SemanticSplitterNodeParser(embed_model=self.embed_model_instance, buffer_size=1, breakpoint_percentile_threshold=85, include_metadata=False)

        img_nodes: list[TextNode] = splitter.get_nodes_from_documents([Document(text=resp)])

        imaged_chunks = {}

        retrieved_images_list = []
        image_scores_list = []
        
        for node in img_nodes:
            text = node.text
            start_idx = node.start_char_idx
            end_idx = node.end_char_idx

            img_preds = self.image_retriever(text)
            images = list(map(lambda p: p.image, img_preds))
            for p in img_preds:
                retrieved_images_list.append(p.image)
                image_scores_list.append(p.score)

            if images:
                imaged_chunks.update({(start_idx, end_idx): images})

        return imaged_chunks, retrieved_images_list, image_scores_list


    def place_images_from_chunks(self, resp: str, imaged_chunks: dict):
        final_resp: str = ""
        last_idx = 0
        generated_images_list = []
        for key, value in imaged_chunks.items():
            if not value: continue

            # logger.debug("Image placement: %s -> %s", key, value)
            temp = f"\n\n![](data:image/png;base64,{image_to_b64(value[0])})\n\n"
            start, end = key

            line_start = resp.rfind("\n", 0, start) + 1

            final_resp += resp[last_idx:line_start] + temp + resp[line_start:end]

            last_idx = end
            generated_images_list.extend(value)

        # Append any remaining text
        final_resp += resp[last_idx:]

        # if no images found, then final response should be the initial response
        if not final_resp.strip():
            final_resp = resp

        return final_resp, generated_images_list


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

