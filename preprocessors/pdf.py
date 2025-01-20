import shutil
import torch
import os
import pypdf
import pymupdf

from datetime import datetime
from utils.embed_utils import get_embedder, compute_similarity_matrix
from typing import List, Optional, Union, Literal, Callable, Any

class PDFPreprocessor:

    def __init__(self, file: Union[str, os.PathLike[str]]) -> None:
        if not isinstance(file, (str, os.PathLike)):
            raise TypeError(f"Expected str, os.PathLike got {type(file).__name__}")

        self.pdf = None
        self.pages: List[pypdf.PageObject] = []

        self.mupdf = pymupdf.open(file)

        self.artifact_loc = ".preprocessor-artifacts"
        dir = os.path.join(os.getcwd(), self.artifact_loc)
        if not os.path.exists(dir):
            os.mkdir(dir)

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

    def _use_deimaged_pdf(self):
        self.pdf = pypdf.PdfReader(os.path.join(os.getcwd(), self.artifact_loc, "artifact.pdf"))
        self.pages = self.pdf.pages

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
        images = []
        rects = []
        file_names = []
        for page in self.mupdf.pages():
            r = []
            for image in page.get_images():
                images.append(self.mupdf.extract_image(image[0]))
                r.append(page.get_image_bbox(image[7]))
                # Delete image from pdf
                page.delete_image(image[0])
            rects.append(r)

        # Save images in out_path folder
        out_path = os.path.abspath(out_path)
        try:
            for image in images:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                fname = f"{file_prefix}{timestamp}.{image['ext']}"
                file_names.append(fname)
                fp = open(os.path.join(out_path, fname), "wb")
                fp.write(image['image'])
                fp.close()
        except KeyError:
            raise RuntimeError(f"unable to extract image data")


        # Replace images with text
        for page_no, page in enumerate(list(self.mupdf.pages())):
            tw = pymupdf.TextWriter(page.rect)
            for img_no, rect in enumerate(rects[page_no]):
                cx = (rect.x0 + rect.x1) // 2
                cy = (rect.y0 + rect.y1) // 2
                tw.append((cx, cy), f"${file_names[img_no]}$")
            tw.write_text(page)

        # Save modified pdf
        self.mupdf.ez_save(os.path.join(os.getcwd(), self.artifact_loc, "artifact.pdf"))


    def deduplicate(self, page_lines: Optional[List[List[str]]] = None, direction: Optional[Literal["up", "down"]] = "down") -> List[List[str]]:
        """
        greedy approach to deduplicate headers and footers
        :return:
        """
        self._use_deimaged_pdf()
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
        else:
            raise ValueError("direction must be 'up' or 'down'")

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
        try:
            self.replace_images("out_images/")
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Skipping image-caption pairing")

        pages = self.deduplicate()
        pages = self.deduplicate(page_lines=pages, direction="up")
        self.cleanup()
        return pages

    def cleanup(self):
        if self.pdf: self.pdf.close()
        self.mupdf.close()
        shutil.rmtree(os.path.join(os.getcwd(), self.artifact_loc))
        # os.rmdir(os.path.join(os.getcwd(), self.artifact_loc))

    def __call__(self, *args, **kwargs):
        return self.forward()
