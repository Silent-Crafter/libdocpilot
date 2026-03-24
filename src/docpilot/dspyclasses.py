import io
import json
import logging
import sys
import time

import dspy
import dspy.streaming
import asyncio
import ollama

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import NodeWithScore, TextNode
from docpilot.signatures import GenerateSearchQuery, GenerateAnswer
from dspy.dsp.utils.utils import deduplicate
from docpilot.utils.image_utils import image_to_b64
from docpilot.utils.embed_utils import get_embedder
from docpilot.utils.logger import (
    get_benchmark_logger,
    BenchmarkTimer,
    build_benchmark_record,
)

from typing import AsyncGenerator, Generator, Union, Optional, List, cast, final
from contextlib import redirect_stdout
from config import Config

logger = logging.getLogger(__name__)
bench_logger = get_benchmark_logger()

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
    def __init__(self, index: VectorStoreIndex, image_index: VectorStoreIndex, num_passages=3):
        super().__init__()

        self.lm: dspy.LM = dspy.settings.lm
        # self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.generate_query = self._query_generator
        self.retrieve = LlamaIndexRMClient(k=num_passages, index=index)
        # self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.generate_answer = self._answer_generator

        self.image_retriever = ImageRetriever(image_index)

        self.message_history: List[dict[str, str]] = []

        self.context = []
        self.files = []

        self.embed_model_instance = get_embedder()[0]


    def _query_generator(self, past_context, question) -> str:
        return self.lm(prompt="""
Generate search query according to the provided question and past context. Respond ONLY with 4-5 precise comma seperated keywords and nothing else
[[ past_context ]]
{pc}

[[ question ]]
{q}""".format(pc=past_context, q=question))[0]


    def _answer_generator(self, context, messages, question) -> str:
        pc = ""
        for idx, ctx in enumerate(context, start=1):
            pc += f"[{idx}]: {ctx}\n"
        return self.lm(prompt="""
Answer question using facts from context and previous conversations only.
If the context is empty or N/A. Always Reply with 'Sorry I cannot assist you with that' regardless of question.
[[ context ]]
{pc}

[[ past_conversation_history ]]
{conv_hist}

[[ question ]]
{q}""".format(pc=pc, conv_hist=messages, q=question))[0]


    def forward(self, question, stream: bool = False, place_images: bool = True):
        timer = BenchmarkTimer()
        context = []
        files = []
        retrieved_images_list = []
        image_scores_list = []
        generated_images_list = []
        resp = ""
        query = question  # fallback in case we bail early
        passages = []
        nodes = None
        history_file = f"{int(time.time())}.txt"

        def serialize_message_history(history: list[dict[str, str]]):
            return '\n'.join(map(lambda d: f"{d['role']}: {d['content']}", history))

        with timer.measure("query_rewrite"):
            query_resp = self.generate_query(past_context=serialize_message_history(self.message_history), question=question)
            # query = query_resp.keywords
            query = query_resp

        # query = question
        yield {"type": "query", "content": query, "status": "Finding files"}

        logger.info("Query: %s", query)

        with timer.measure("retrieval"):
            nodes = self.retrieve(query)
        passages = list(reversed(nodes.passages))

        files = deduplicate(files + nodes.files)
        context = deduplicate(context + passages)

        self.context = deduplicate(context + self.context)
        self.files = deduplicate(files + self.files)

        yield {"type": "files", "content": self.files, "status": "Generating answer"}

        with timer.measure("generation"):
            if False:
                ... 
                # streamed_generate_answer = dspy.streamify(
                #     self.generate_answer,
                #     stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
                #     async_streaming=False
                # )
                #
                # resp_generator: Generator = streamed_generate_answer(context=self.context, question=question, messages=self.format_history())
                #
                # for chunk in resp_generator:
                #     if isinstance(chunk, dspy.Prediction):
                #         resp = chunk.answer
                #         yield {
                #             "type": "answer",
                #             "content": chunk.answer,
                #             "status": "Inserting images"
                #         }
                #         break
                #
                #     yield {
                #         "type": "streaming_answer",
                #         "content": chunk.chunk,
                #         "status": "Streaming"
                #     }

            else:
                prediction = self.generate_answer(
                    context=self.context,
                    question=question,
                    messages=self.format_history()
                )
                # resp = prediction.answer
                resp = prediction

                yield {"type": "answer", "content": resp, "status": "Inserting images"}

        if place_images:
            with timer.measure("image_retrieval"):
                imaged_chunks, ril, isl = self.create_image_chunks(resp)
                retrieved_images_list.extend(ril)
                image_scores_list.extend(isl)

            with timer.measure("image_placement"):
                final_resp, gil  = self.place_images_from_chunks(resp, imaged_chunks)
                generated_images_list.extend(gil)
                yield {"type": "answer_with_images", "content": final_resp, "status": "DONE"}

        lm = dspy.settings.lm
        model_name = getattr(lm, "model", "") if lm else ""

        lm_history = []

        for history in lm.history:
            messages = history.get('messages')
            response = history.get('response')
            kwargs = history.get('kwargs')
            usage = history.get('usage')

            usage['completion_tokens_details'] = str(usage.get('completion_tokens_details', ''))

            resp = ""
            for choice in getattr(response, 'choices', []):
                resp += '\n' + choice.message.content + '\n'

            lm_history.append({
                "messages": messages,
                "response": resp,
                "kwargs": kwargs,
                "usage": usage
            })

        record = build_benchmark_record(
            query=question,
            rewritten_query=query,
            retrieved_docs=[p[:80] for p in passages],
            retrieval_scores=list(nodes.scores) if nodes and hasattr(nodes, "scores") else [],
            source_files=list(self.files),
            retrieved_images=retrieved_images_list,
            image_retrieval_scores=image_scores_list,
            generated_images=generated_images_list,
            answer=resp,
            model=model_name,
            embed_model=Config.embed_model,
            num_hops=1,
            context_chunks_used=len(self.context),
            streaming=stream,
            latencies=timer.get_latencies(),
            lm_history=lm_history
        )

        bench_logger.info(json.dumps(record))
        logger.info("Benchmark record emitted for query: %s", question)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            dspy.inspect_history()

        with open(history_file, 'w') as f:
            f.write(buffer.getvalue())

        self.update_message_history([
            {"role": "user", "content": question},
            {"role": "assistant", "content": resp},
        ])

        self.lm.history.clear()

        yield {"type": "finalization", "content": None, "status": "DONE"}


    def create_image_chunks(self, resp):
        splitter = SemanticSplitterNodeParser(embed_model=self.embed_model_instance, buffer_size=1, breakpoint_percentile_threshold=85, include_metadata=False)

        img_nodes: list[TextNode] = cast(List[TextNode], splitter.get_nodes_from_documents([Document(text=resp)]))

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

