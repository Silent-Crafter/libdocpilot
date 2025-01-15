import torch

from os import PathLike
from pypdf import PdfReader, PageObject

from utils.embed_utils import get_embedder, compute_similarity_matrix

from typing import List, Optional, Union, Literal

class PDFPreprocessor:

    def __init__(self, file: Union[str, PathLike[str], PdfReader]) -> None:
        if not isinstance(file, (str, PathLike, PdfReader)):
            raise TypeError(f"Expected str, PathLike, PdfReader got {type(file).__name__}")

        if not isinstance(file, PdfReader):
            self.pdf = PdfReader(file)
        else:
            self.pdf = file

        self.pages: List[PageObject] = self.pdf.pages

    def _pages_into_lines(self, strip_rotated: Optional[bool] = False) -> List[List[str]]:
        if not self.pdf.pages:
            raise RuntimeError("No pages found")

        lines: List[List[str]] = []
        for page in self.pages:
            text = page.extract_text(
                extraction_mode='layout',
                layout_mode_strip_rotated=strip_rotated,
                layout_mode_space_vertically=False,
            )

            lines.append(text.splitlines())

        return lines

    def deduplicate(self, page_lines: Optional[List[List[str]]] = None, direction: Optional[Literal["up", "down"]] = "down") -> List[List[str]]:
        """
        Naive, loose approach to deduplicate headers and footers
        :return:
        """
        lines = self._pages_into_lines(strip_rotated=True) if not page_lines else page_lines

        _, embed = get_embedder("hf/ibm-granite/granite-embedding-278m-multilingual")

        # Remove blank lines if any
        lines = [
            [line for line in page if line.strip()] for page in lines
        ]

        no_of_pages = len(lines)
        new_pages: List[List[str]] = [
            [] for _ in range(no_of_pages)
        ]

        # no. of lines on every page
        lens = list(map(len, lines))
        max_len = max(lens)

        def condition(cnt: int) -> bool:
            return cnt < max_len if direction == "down" else abs(cnt) <= max_len

        if direction == "up":
            counter = -1
        elif direction == "down":
            counter = 0

        while condition(counter):
            embeddings: List[torch.Tensor] = []
            lines_ = []

            # Compute embedding of nth line of every page
            for i in range(no_of_pages):
                if counter < lens[i] and direction == "down":
                    line = lines[i][counter]
                elif abs(counter) <= lens[i] and direction == "up":
                    line = lines[i][counter+lens[i]]
                else:
                    line = ''

                lines_.append(line)

            # Compute embedding of each line
            embeddings.extend([embed(line) for line in lines_])

            # Compute a matrix which contains computed similarity of any two given lines
            similarity_matrix = compute_similarity_matrix(no_of_pages, no_of_pages, embeddings)

            # Remove similar lines and store the results page by page
            discard_index = set(
                j+i+1
                for i, s1 in enumerate(similarity_matrix)
                for j, s2 in enumerate(s1[i+1:])
                if s2 > 0.86
            )

            # Merge all the lines back into pages
            for i in range(no_of_pages):
                if i not in discard_index:
                    if direction == "up":
                        new_pages[i].insert(0,lines_[i])
                    elif direction == "down":
                        new_pages[i].append(lines_[i])

            if direction == "up":
                counter -= 1
            else:
                counter += 1

        return new_pages


    def forward(self):
        pages = self.deduplicate()
        pages = self.deduplicate(page_lines=pages, direction="up")
        return pages

    def __call__(self, *args, **kwargs):
        return self.forward()
