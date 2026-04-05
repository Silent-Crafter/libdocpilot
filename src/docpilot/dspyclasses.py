import logging
import os
import dspy

from pathlib import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter, SimpleNodeParser
from llama_index.core.schema import NodeWithScore, TextNode
from docpilot.parsers import CustomPDFReader
from docpilot.signatures import AnswerPrompt, QueryPrompt, build_prompt_from_signature
from dspy.dsp.utils.utils import deduplicate
from docpilot.utils.image_utils import image_to_b64, mappings_to_llamaindex_document
from docpilot.utils.embed_utils import Embedder
from docpilot.lm import DspyLMWrapper
from collections import defaultdict

from typing import Union, Optional, List, Callable, Any
from config import Config
logger = logging.getLogger(__name__)

class ImageRanker:
    def __init__(self):
        self.inverted_map: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def set(self, line: str, image: str, score: int, start_idx: int, end_idx: int):
        self.inverted_map[image].append({"line": line, "score": score, "start_idx": start_idx, "end_idx": end_idx})

    def rank(self, top_k: int = 1) -> dict[str, list[dict[str, Any]]]:
        # First sort by score
        for image in self.inverted_map.keys():
            self.inverted_map[image] = sorted(self.inverted_map[image], key=lambda d: d['score'])

        # Construct and return a normal line -> image map
        image_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for image, data in self.inverted_map.items():
            if len(image_map[data[0]['line']]) < top_k:
                image_map[data[0]['line']].append({
                    "image": image,
                    "start_idx": data[0]["start_idx"],
                    "end_idx": data[0]["end_idx"],
                    "score": data[0]["score"],
                })

        return image_map

class LlamaIndexRMClient(dspy.Retrieve):
    def __init__(self, index: VectorStoreIndex, k: int = 3):
        super().__init__(k=k)
        self.index: VectorStoreIndex = index
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

    def update_index(self, nodes: list):
        self.index.insert_nodes(nodes)
        self.retriever = self.index.as_retriever()

    def __call__(self, *args, **kwargs) -> dspy.Prediction:
        return self.forward(*args, **kwargs)


class ImageRetriever(dspy.Retrieve):
    def __init__(self, image_index: VectorStoreIndex):
        super().__init__()
        self.index = image_index
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

    def update_labels(self, new_labels: dict):
        docs = mappings_to_llamaindex_document(new_labels, 'out_image')
        nodes = SimpleNodeParser().get_nodes_from_documents(docs)
        self.index.insert_nodes(nodes)
        self.retriever = self.index.as_retriever()

    def __call__(self, *args, **kwargs) -> List[dspy.Prediction]: return self.forward(*args, **kwargs)


class MultiHopRAG(dspy.Module):
    embed_model_instance: Optional[HuggingFaceEmbedding] = None
    retrieve: Optional[LlamaIndexRMClient] = None
    image_retriever: Optional[ImageRetriever] = None

    def __init__(self, index: VectorStoreIndex, image_index: VectorStoreIndex, num_passages=3):
        super().__init__()

        if self.__class__.retrieve is None:
            self.__class__.retrieve = LlamaIndexRMClient(k=num_passages, index=index)

        if self.__class__.image_retriever is None:
            self.__class__.image_retriever = ImageRetriever(image_index)

        if self.__class__.embed_model_instance is None:
            self.__class__.embed_model_instance = Embedder.get_embedder()

        self.generate_query = self.__resp_generator(QueryPrompt)
        self.generate_answer = self.__resp_generator(AnswerPrompt)

        self.message_history: List[dict[str, str]] = []
        self.message_history_with_images: List[dict[str, str]] = []
        self.ranker = ImageRanker()

        self.context = []
        self.files = []

    def __resp_generator(self, signature) -> Callable[..., Any]:
        def __generator(**kwargs):
            stream = kwargs.pop('stream', False)
            prompt = build_prompt_from_signature(signature, kwargs)
            lm: DspyLMWrapper | None  = dspy.settings.get('lm')
            if lm is None:
                raise RuntimeError("Please configre an lm")

            return lm(prompt=prompt, stream=stream, **kwargs)

        return __generator

    def forward(self, question, stream: bool = False):
        context = []
        files = []
        resp = ""
        passages = []
        nodes = None

        def serialize_message_history(history: list[dict[str, str]]):
            return '\n'.join(map(lambda d: f"{d['role']}: {d['content']}", history))

        query_resp = self.generate_query(past_context=serialize_message_history(self.message_history), question=question)
        # query = query_resp.keywords
        query = query_resp

        yield {"type": "query", "content": query, "status": "Finding files"}

        logger.info("Query: %s", query)
        nodes = self.__class__.retrieve(query)
        passages = list(reversed(nodes.passages))

        files = deduplicate(files + nodes.files)
        context = deduplicate(context + passages)

        self.context = deduplicate(context + self.context)
        self.files = deduplicate(files + self.files)

        yield {"type": "files", "content": self.files, "status": "Generating answer"}

        if stream:
            accumulated = ""
            for chunk in self.generate_answer(context=self.context, question=question, messages=self.format_history(), stream=True):
                accumulated += chunk
                yield {
                    "type": "streaming_answer",
                    "content": chunk,
                    "status": "Streaming"
                }

            resp = accumulated

        else:
            prediction = self.generate_answer(
                context=self.context,
                question=question,
                messages=self.format_history()
            )
            resp = prediction

        yield {"type": "answer", "content": resp, "status": "Inserting images"}

        # imaged_chunks, *_ = self.create_image_chunks(resp)
        imaged_chunks = self.__rank_images(resp)
        final_resp, *_  = self.place_images_from_chunks(resp, imaged_chunks)
        yield {"type": "answer_with_images", "content": final_resp, "status": "DONE"}

        self.update_message_history([
            {"role": "user", "content": question},
            {"role": "assistant", "content": resp},
        ])
        self.update_message_history_with_images([
            {"role": "user", "content": question},
            {"role": "assistant", "content": final_resp},
        ])
        yield {"type": "finalization", "content": None, "status": "DONE"}


    def create_image_chunks(self, resp):
        splitter = SemanticSplitterNodeParser(embed_model=self.__class__.embed_model_instance, buffer_size=1, breakpoint_percentile_threshold=85, include_metadata=False)

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

    def __rank_images(self, actual_answer):
        ranker = ImageRanker()
        start_idx = 0
        end_idx = -1
        for line in actual_answer.splitlines():
            if not line.strip():
                start_idx += len(line) + 1
                continue

            end_idx = start_idx + len(line)
            image_nodes = self.__class__.image_retriever(line)
            
            for image in image_nodes:
                ranker.set(line, image.image, image.score, start_idx, end_idx)

            start_idx = end_idx + 1

        ranked = ranker.rank()

        # Reshape from {line: [{image, start_idx, end_idx, score}, ...]}
        # into {(start_idx, end_idx): [image_filename, ...]}
        # which is what place_images_from_chunks expects
        imaged_chunks: dict[tuple[int, int], list[str]] = {}
        for line_text, entries in ranked.items():
            for entry in entries:
                key = (entry["start_idx"], entry["end_idx"])
                imaged_chunks.setdefault(key, []).append(entry["image"])

        return imaged_chunks

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

    def update_message_history_with_images(self, messages: Union[List[dict[str, str]], str]):
        if isinstance(messages, str):
            raise NotImplementedError

        self.message_history_with_images.extend(messages)


    def format_history(self) -> str:
        if not self.message_history:
            return ""

        messages = "\n".join(map(
            lambda m: m["role"].title() + ": " + m["content"],
            self.message_history
        ))

        return messages


    def add_new_document(self, file_path: str | Path):
        if self.__class__.retrieve is None:
            raise RuntimeError(f"Instantiate {self.__class__.__qualname__} first")

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        image_docs: list[Document] = []

        if ext == '.pdf':
            reader = CustomPDFReader()
            docs = reader.load_data(file_path)
            image_docs = reader.image_documents
        else:
            docs = SimpleDirectoryReader(input_files=[file_path]).load_data(show_progress=True)

        # Index text documents
        splitter = SemanticSplitterNodeParser(buffer_size=3, embed_model=self.__class__.embed_model_instance)
        nodes = splitter.get_nodes_from_documents(docs)
        self.__class__.retrieve.update_index(nodes)

        # Index image label documents into the image index
        if image_docs and self.__class__.image_retriever is not None:
            image_nodes = SimpleNodeParser().get_nodes_from_documents(image_docs)
            self.__class__.image_retriever.index.insert_nodes(image_nodes)
            self.__class__.image_retriever.retriever = self.__class__.image_retriever.index.as_retriever()

    def delete_document(self, filename: str, uri: str, text_table: str, image_table: str):
        """
        Delete a document and all its associated data:
        1. Remove embedding rows from data_images (by source_file)
        2. Delete associated image files from disk
        3. Remove embedding rows from data_items (by file_name)
        4. Rebuild in-memory indexes from the vector stores

        :param filename: The basename of the document file (e.g. 'Machine Learning.pdf')
        :param uri: Postgres connection URI
        :param text_table: The embeddings table for text (e.g. 'data_items')
        :param image_table: The embeddings table for images (e.g. 'data_images')
        """
        import psycopg2

        conn = psycopg2.connect(uri)
        cursor = conn.cursor()

        try:
            # 1. Find image file paths from data_images before deleting
            cursor.execute(
                f"SELECT DISTINCT metadata_->>'file_name' FROM {image_table} "
                f"WHERE metadata_->>'source_file' = %s",
                (filename,)
            )
            image_paths = [row[0] for row in cursor.fetchall() if row[0]]

            # 2. Delete rows from data_images
            cursor.execute(
                f"DELETE FROM {image_table} WHERE metadata_->>'source_file' = %s",
                (filename,)
            )
            deleted_images = cursor.rowcount
            logger.info("Deleted %d rows from %s for source_file=%s", deleted_images, image_table, filename)

            # 3. Delete rows from data_items
            cursor.execute(
                f"DELETE FROM {text_table} WHERE metadata_->>'file_name' = %s",
                (filename,)
            )
            deleted_texts = cursor.rowcount
            logger.info("Deleted %d rows from %s for file_name=%s", deleted_texts, text_table, filename)

            conn.commit()

            # 4. Delete image files from disk
            for img_path in image_paths:
                try:
                    if os.path.isfile(img_path):
                        os.remove(img_path)
                        logger.info("Deleted image file: %s", img_path)
                except OSError as e:
                    logger.warning("Could not delete image file %s: %s", img_path, e)

            # 5. Rebuild in-memory indexes from the vector stores
            self._rebuild_indexes(uri, text_table, image_table)

        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def _rebuild_indexes(self, uri: str, text_table: str, image_table: str):
        """Rebuild VectorStoreIndex instances from PG and reassign to retrievers."""
        from docpilot.utils.llama_utils import get_vector_storage_context

        embed_model = self.__class__.embed_model_instance

        # Rebuild text index
        _, text_vs = get_vector_storage_context(uri, text_table, perform_setup=False)
        text_index = VectorStoreIndex.from_vector_store(
            vector_store=text_vs,
            embed_model=embed_model,
        )
        if self.__class__.retrieve is not None:
            self.__class__.retrieve.index = text_index
            self.__class__.retrieve.retriever = text_index.as_retriever(
                similarity_top_k=self.__class__.retrieve.k
            )

        # Rebuild image index
        _, image_vs = get_vector_storage_context(uri, image_table, perform_setup=False)
        image_index = VectorStoreIndex.from_vector_store(
            vector_store=image_vs,
            embed_model=embed_model,
        )
        if self.__class__.image_retriever is not None:
            self.__class__.image_retriever.index = image_index
            self.__class__.image_retriever.retriever = image_index.as_retriever()


def configure_llm(model: str, cache: bool, base_url: Optional[str] = None, **kwargs):
    if base_url:
        lm = DspyLMWrapper(model=model, base_url=base_url, cache=cache, **kwargs)
    else:
        lm = DspyLMWrapper(model=model, cache=cache, **kwargs)
    dspy.settings.configure(lm=lm)
    return lm

