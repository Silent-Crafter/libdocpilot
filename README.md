# libdocpilot

**libdocpilot** is a Python library that provides an enhanced, multi-hop RAG (Retrieval-Augmented Generation) pipeline for chatting with documents. It goes beyond plain text retrieval by automatically extracting images from documents, understanding their surrounding context, and inserting relevant images inline into the generated answer.

Built on top of [LlamaIndex](https://github.com/run-llama/llama_index) and [DSPy](https://github.com/stanfordnlp/dspy), it supports PDF, XLSX, and other document formats, with a PostgreSQL (`pgvector`) backend for persistent vector storage.

---

## Features

- 📄 **Rich document parsing** — Extracts text, tables, and images from PDFs with bounding-box awareness. Also handles XLSX files.
- 🖼️ **Image-context retrieval** — Maps each extracted image to its surrounding text using either *sequential* (single-column) or *spatial* (multi-column / floating) context extraction modes.
- 🔍 **Multi-hop retrieval** — Rewrites the user query into precise search keywords before retrieval to improve relevance.
- 💬 **Conversational memory** — Maintains per-session message history so follow-up questions are answered with context.
- 🗄️ **Persistent vector store** — Uses PostgreSQL + `pgvector` (via LlamaIndex's `PGVectorStore`) with HNSW indexing. Falls back to an in-memory index if Postgres is unavailable.
- ➕ **Incremental indexing** — Only re-embeds documents that are not already present in the vector store.
- 🗑️ **Document deletion** — Removes a document's text embeddings, image embeddings, and on-disk image files, then rebuilds the in-memory indexes automatically.
- 🤖 **LLM-agnostic** — Works with any Ollama-served model or any LiteLLM-compatible provider through DSPy.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         libdocpilot                              │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ PDFPreproc. │    │ CustomPDF/   │    │   load_docs()      │  │
│  │  get_elems  │───▶│ XLSXReader   │───▶│ (llama_utils)      │  │
│  │  get_image  │    │ (parsers.py) │    │ Skips indexed docs │  │
│  │  context    │    └──────────────┘    └────────┬───────────┘  │
│  └─────────────┘                                 │              │
│                                           ┌──────▼──────┐       │
│                                           │ get_vector_ │       │
│                                           │ store_index │       │
│                                           │  (PGVector/ │       │
│                                           │  in-memory) │       │
│                                           └──────┬──────┘       │
│                                                  │              │
│  ┌───────────────────────────────────────────────▼───────────┐  │
│  │                     MultiHopRAG                           │  │
│  │  LlamaIndexRMClient (text)   ImageRetriever (images)      │  │
│  │  QueryPrompt rewrite ──▶ retrieve ──▶ AnswerPrompt        │  │
│  │  ImageRanker ──▶ place_images_from_chunks ──▶ yield       │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

`MultiHopRAG.forward()` is a **generator** that yields intermediate status events so you can stream results to a UI or CLI as they arrive:

| Event type | Content |
|---|---|
| `query` | Rewritten search keywords |
| `files` | Source filenames retrieved |
| `answer` | Plain-text LLM answer |
| `answer_with_images` | Answer with inline base-64 images |
| `streaming_answer` | Individual streamed text chunks (stream mode) |
| `finalization` | Signal that the turn is complete |

---

## Requirements

- Python ≥ 3.12
- [Ollama](https://ollama.com/) running locally (default: `http://localhost:11434`)
- PostgreSQL with `pgvector` extension (recommended; in-memory fallback available)

---

## Installation

### From source

```shell
git clone https://github.com/Silent-Crafter/libdocpilot
cd libdocpilot
pip install .
```

### With `uv` (recommended)

```shell
git clone https://github.com/Silent-Crafter/libdocpilot
cd libdocpilot
uv sync
```

---

## Configuration

Copy `config.py` into your project root (or inherit from it) and fill in the values:

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    PG_CONNECTION_URI  = os.getenv("PG_CONNECTION_URI")          # e.g. "postgresql://user:pass@localhost:5432/docpilot"
    embed_table        = "data_items"                             # PG table for text embeddings (must be prefixed with "data_")
    embed_model        = "ibm-granite/granite-embedding-278m-multilingual"  # HuggingFace embedding model
    ollama_model       = "dolphin3"                               # Ollama model tag
    ollama_url         = "http://localhost:11434"                 # Ollama base URL
    document_dir       = "data"                                   # Directory containing your documents
```

Set `PG_CONNECTION_URI` in a `.env` file or as an environment variable:

```
PG_CONNECTION_URI=postgresql://user:password@localhost:5432/docpilot
```

---

## Quick Start

```python
import dspy
from docpilot.dspyclasses import MultiHopRAG
from docpilot.utils.llama_utils import get_vector_store_index, load_docs
from config import Config as config

# 1. Load and embed documents (skips already-indexed ones automatically)
docs, image_docs, image_mappings = load_docs(
    config.document_dir,
    config.PG_CONNECTION_URI,
    config.embed_table,
)

# 2. Build vector indexes
index       = get_vector_store_index(docs,       config.embed_model, embeddings_table=config.embed_table,  uri=config.PG_CONNECTION_URI)
image_index = get_vector_store_index(image_docs, config.embed_model, embeddings_table="data_images",       uri=config.PG_CONNECTION_URI)

# 3. Configure DSPy LLM
dspy.settings.configure(lm=dspy.LM(
    model=f"ollama/{config.ollama_model}",
    base_url=config.ollama_url,
    cache=False,
))

# 4. Create the RAG module
rag = MultiHopRAG(index=index, image_index=image_index, num_passages=3)

# 5. Ask a question
for event in rag.forward("What is gradient descent?"):
    if event["type"] == "answer":
        print(event["content"])
    elif event["type"] == "answer_with_images":
        # Markdown string with base-64 images embedded
        with open("answer.html", "w") as f:
            f.write(event["content"])
```

---

<!-- ## API Reference

### `load_docs(doc_dir, uri, embedding_table, reindex=False, use_vlm=False)`

Scans `doc_dir` for documents, skips files already present in the vector store, and returns:

- `documents` — LlamaIndex `Document` objects for text content
- `image_documents` — `Document` objects whose text is the surrounding context of each extracted image
- `image_mappings` — raw `{image_path: context_text}` dict (useful for debugging)

Supported file types: `.pdf`, `.xlsx`, and anything handled by LlamaIndex's `SimpleDirectoryReader`.

---

### `get_vector_store_index(documents, embed_model, uri, embeddings_table, reindex=False, embed_dim=768)`

Builds or updates a `VectorStoreIndex` backed by `PGVectorStore`.

- If Postgres is unreachable, automatically falls back to an in-memory index.
- Set `reindex=True` to truncate the table and rebuild from scratch.
- Uses **semantic chunking** (`SemanticSplitterNodeParser`) with a 90th-percentile breakpoint threshold.

---

### `MultiHopRAG(index, image_index, num_passages=3)`

The main RAG module. Class-level attributes (`retrieve`, `image_retriever`, `embed_model_instance`) are shared across instances, so multiple `MultiHopRAG` objects reuse the same retrievers.

#### `forward(question, stream=False)`

Generator. Yields event dicts — see the table in the [Architecture](#architecture) section.

#### `add_new_document(file_path)`

Parses a single file, semantically chunks it, and inserts its nodes into the live indexes without restarting.

```python
rag.add_new_document("data/new_report.pdf")
```

#### `delete_document(filename, uri, text_table, image_table)`

Removes a document from the system:

1. Fetches image file paths from the image embeddings table.
2. Deletes rows from the image embeddings table (`source_file` filter).
3. Deletes rows from the text embeddings table (`file_name` filter).
4. Deletes image files from disk.
5. Rebuilds in-memory indexes from the updated vector stores.

```python
rag.delete_document(
    filename="Machine Learning.pdf",
    uri=config.PG_CONNECTION_URI,
    text_table="data_items",
    image_table="data_images",
)
```

---

### `PDFPreprocessor(file_path)`

Low-level PDF element extractor built on [PyMuPDF](https://pymupdf.readthedocs.io/).

| Method | Description |
|---|---|
| `get_elements()` | Returns a list of `{type, content, bbox, page}` dicts for all text, image, and table blocks. |
| `get_image_context(method, context_window, max_distance, elements, use_vlm)` | Returns `{image_path: surrounding_text}`. `method` is `'sequential'` or `'spatial'`. |
| `get_image_context_sequential(elements, context_window)` | Collects the N nearest text blocks before/after each image in document order. Best for single-column PDFs. |
| `get_image_context_spatial(elements, context_window, max_distance)` | Finds nearest same-page text/table blocks by bounding-box distance. Best for multi-column layouts. |
| `close()` | Releases the underlying PDF document handle. |

---

### `CustomPDFReader` / `CustomXLSXReader`

LlamaIndex `BaseReader` subclasses used by `SimpleDirectoryReader`.

- `CustomPDFReader` delegates to `PDFPreprocessor` and populates `reader.image_documents` and `reader.image_mappings` as side-channels.
- `CustomXLSXReader` reads all sheets, joins columns with a configurable delimiter, and optionally concatenates rows into a single `Document` per sheet.

--- -->

## Multi-Session Usage

Each `MultiHopRAG` instance maintains its own `message_history`, making it straightforward to run isolated concurrent sessions — simply instantiate one object per session:

```python
session_a = MultiHopRAG(index=index, image_index=image_index)
session_b = MultiHopRAG(index=index, image_index=image_index)
```

The underlying retrieval components are shared (class-level) while the conversation state is per-instance.

---

## Logging

libdocpilot uses Python's standard `logging` module throughout. To enable output:

```python
from docpilot.utils.logger import setup_logging
setup_logging()  # Configures a stdout handler + optional metrics file
```

Set the `LOG_LEVEL` environment variable to control verbosity (`DEBUG`, `INFO`, `WARNING`, etc.).

---

## Related Projects

- [docpilot-api](https://github.com/Silent-Crafter/docpilot-api) — Flask REST API that wraps this library

---

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
