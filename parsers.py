import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import openpyxl
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import os

import re

class CustomXLSXReader(BaseReader):
    def __init__(
            self,
            *args: Any,
            concat_rows: bool = True,
            col_joiner: str = ",",
            row_joiner: str = "\n",
            pandas_config: dict = {},
            **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config

    def load_data(
            self,
            file: Path,
            extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Parse file."""

        wb = openpyxl.load_workbook(file)

        documents: list[Document] = []
        for ws in wb.sheetnames:
            df = pd.read_excel(file, sheet_name=ws, **self._pandas_config)

            sys.stdout.flush()
            text_list = [" ".join(df.columns.astype(str))]  # Concat headers
            text_list += (
                df.astype(str)
                .apply(lambda row: self._col_joiner.join(row.values), axis=1)
                .tolist()
            )

            metadata = {"filename": file.name, "extension": file.suffix}
            if extra_info:
                metadata.update(extra_info)

            if self._concat_rows:
                documents.append(Document(
                    text=self._row_joiner.join(text_list),
                    metadata=metadata,
                ))
            else:
                documents.extend([
                    Document(
                        text=text,
                        metadata=metadata,
                    )
                    for text in text_list
                ])


        return documents


class CustomPDFReader(BaseReader):
    """PDF parser."""

    def __init__(self, return_full_document: Optional[bool] = False) -> None:
        """
        Initialize PDFReader.
        """
        self.return_full_document = return_full_document

    def load_data(
            self,
            file: Path,
            extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Parse file."""
        if not isinstance(file, Path):
            file = Path(file)

        metadata = {"file_name": file.name}

        docs = []

        os.system(f"pdftotext -layout -nodiag '{file.as_posix()}'")
        txt_file_name = file.name.replace(file.suffix, ".txt")
        print('Reading', txt_file_name)
        text = ''
        with open("data/"+txt_file_name, "r", encoding="utf-8") as f:
            text = f.read()

        # Small optimization to improve table recognition capability for an llm
        # Also helps to reduce data loss while chunking as there will be far less characters
        text = re.sub(" {6}", " ", text)

        os.remove(file.resolve().as_posix().replace(file.suffix, ".txt"))

        # Join text extracted from each page
        docs.append(Document(text=text, metadata=metadata))

        return docs

if __name__ == '__main__':
    from llama_index.core import SimpleDirectoryReader

    # documents = SimpleDirectoryReader(
    #     "data",
    #     file_extractor={".xlsx": CustomXLSXReader(), ".pdf": CustomPDFReader()},
    #     required_exts=['.pdf']
    # ).load_data()

    documents = CustomPDFReader().load_data(Path("data/Control Plan - 20. winding CP rev-28.pdf"))

    print(documents[0].text)
