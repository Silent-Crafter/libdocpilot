import shutil
import torch
import os
import pypdf

import pymupdf

from datetime import datetime
from docpilot.utils.embed_utils import get_embedder, compute_similarity_matrix
from typing import List, Optional, Union, Literal

from docpilot.notlogging.notlogger import NotALogger

logger = NotALogger(__name__)
logger.enabled = False

class PDFPreprocessor:

    def __init__(self, file: Optional[Union[str, os.PathLike[str]]] = None) -> None:
        if file and not isinstance(file, (str, os.PathLike)):
            raise TypeError(f"Expected str, os.PathLike got {type(file).__name__}")

        self.file = file
        self.pdf = None
        self.pages: List[pypdf.PageObject] = []

        self.mupdf = pymupdf.open(file) if file else None
        _, self.embed = get_embedder()

    def _pages_into_lines(self, strip_rotated: Optional[bool] = False) -> List[List[str]]:
        if not self.pdf.pages:
            raise RuntimeError("No pages found")

        if not self.pages:
            self.pages = self.pdf.pages

        lines: List[List[str]] = []
        for page in self.pages:
            text = page.extract_text(
                extraction_mode='layout',
                layout_mode_strip_rotated=strip_rotated,
                layout_mode_space_vertically=False,
            )

            lines.append(text.splitlines())

        return lines

    def replace_images(
        self,
        out_path: Optional[Union[str, os.PathLike[str]]] = None,
        prefix: Optional[str] = None,
    ):
        if not isinstance(out_path, (str, os.PathLike)):
            raise TypeError(f"Expected str, os.PathLike, got {type(out_path).__name__}")

        txt_prefix = prefix if prefix else "img"
        file_prefix = txt_prefix

        # Extract images with their bounding boxes
        image_xrefs = []
        rects = []
        file_names = []
        for page in self.mupdf.pages():
            r = []
            for image in page.get_images():
                image_xrefs.append(image[0])
                r.append(page.get_image_bbox(image[7]))
                # Delete image from pdf
                # page.delete_image(image[0])
            rects.append(r)

        logger.info(f"Found {len(image_xrefs)} images")

        # TODO: USE PYPDF FOR EXTRACTING IMAGES
        #       DROP PYMUPDF DEPENDENCY
        # Save images in out_path folder
        out_path = os.path.abspath(out_path)
        try:
            for xref in image_xrefs:
                image = self.mupdf.extract_image(xref)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                fname = f"{file_prefix}-{timestamp}-{xref}.{image['ext']}"
                file_names.append(fname)
                with open(os.path.join(out_path, fname), "wb") as fp:
                    fp.write(image['image'])
        except Exception as e:
            raise RuntimeError(f"Error while extracting image: {e}")


    def deduplicate(
            self,
            page_lines: Optional[List[List[str]]] = None,
            direction: Optional[Literal["up", "down"]] = "down"
    ) -> List[List[str]]:
        """
        greedy approach to deduplicate headers and footers
        :return:
        """
        if direction not in ["up", "down"]:
            raise ValueError("direction must be 'up' or 'down'")

        lines = self._pages_into_lines(strip_rotated=True) if not page_lines else page_lines

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

        counter = 0 if direction == "down" else -1

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
            embeddings.extend([self.embed(line) for line in lines_])

            # Compute a matrix which contains computed similarity of any two given lines
            similarity_matrix = compute_similarity_matrix(no_of_pages, no_of_pages, embeddings)

            # Remove similar lines and store the results page by page
            discard_index = set(
                j+i+1
                for i, s1 in enumerate(similarity_matrix)
                for j, s2 in enumerate(s1[i+1:])
                if s2 > 0.9
            )

            # Merge all the lines back into pages
            for i in range(no_of_pages):
                if i in discard_index: continue

                if direction == "up":
                    new_pages[i].insert(0,lines_[i])
                else:
                    new_pages[i].append(lines_[i])

            counter = counter + (1 if direction == "down" else -1)

        return new_pages

    def open(self, file: Union[str, os.PathLike[str]]):
        self.file = file
        self.mupdf = pymupdf.open(file)
        self.pdf = pypdf.PdfReader(self.file)
        self.pages = self.pdf.pages

    def close(self):
        self.file = None
        if not self.mupdf.is_closed: self.mupdf.close()
        if self.pdf: 
            self.pdf.close()
            self.pages = []

