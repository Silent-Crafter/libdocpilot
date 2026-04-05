import io
import math
import hashlib
import os
import fitz
import logging
import ollama

from collections import defaultdict
from typing import List, Literal, Optional, Union, Dict, Tuple, Iterable, Any
from PIL import Image
from sqlalchemy.engine.result import ResultInternal

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PDFPreprocessor:
    """A preprocessor for extracting elements from PDF documents.
    
    This class handles extraction of text, images, and tables from PDF files,
    with support for image processing including soft masks and transparency handling.
    """

    def __init__(self, file_path: Optional[Union[str, os.PathLike[str]]]) -> None:
        if not isinstance(file_path, (str, os.PathLike)):
            raise TypeError(f"Expected str, os.PathLike got {type(file_path).__name__}")

        self.file_path = file_path

        # weird type hint fix
        self.doc: fitz.Document = fitz.open(file_path)
        self.img_dir = "out_images/"
        os.makedirs(self.img_dir, exist_ok=True)

    def __get_overlap_ratio(self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
        x0 = max(bbox1[0], bbox2[0])
        y0 = max(bbox1[1], bbox2[1])
        x1 = min(bbox1[2], bbox2[2])
        y1 = min(bbox1[3], bbox2[3])

        if(x0>=x1 or y0>=y1):
            return 0.0
        
        inter_area=(x1 - x0) * (y1 - y0)
        bbox1_area=(bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

        if(bbox1_area==0):
            return 0.0        
        return inter_area/bbox1_area
        
    def __extract_and_apply_smask(self, raw_bytes, smask_xref) -> bytes:
        """Extract and apply given image raw_bytes with its soft mask (SMask) from PDF.
        
        Args:
            raw_bytes: Image item tuple containing xref and optional smask_xref.
            smask_xref: Image item tuple containing xref and optional smask_xref.
            
        Returns:
            1) Tuple of (image_bytes, image_format) where image_bytes is the processed image
            and image_format is the file extension (e.g., 'png').
            2) None if extraction fails.
        """
        
        if smask_xref == 0 or raw_bytes is None:
            return raw_bytes

        logging.debug(f"smask={smask_xref} detected : applying manual alpha")

        # Load color image
        colour_img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")

        # Load SMask
        smask_ei = self.doc.extract_image(smask_xref)
        mask_img = Image.open(io.BytesIO(smask_ei["image"])).convert("L")

        # Resize mask accordingly
        if mask_img.size != colour_img.size:
            mask_img = mask_img.resize(colour_img.size)

        # Apply mask as alpha channel
        colour_img.putalpha(mask_img)

        # Convert back to bytes (PNG to preserve transparency)
        buf = io.BytesIO()
        colour_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        return img_bytes
            
    def __process_image(self, img_bytes: bytes) -> Union[Image.Image, None]:
        """Process an image by converting to RGBA, handling transparency, and saving.
        
        Args:
            img_bytes: Raw image bytes to process.
            
        Returns:
            1) Path to the saved processed image.
            2) None if processing fails.
        """    
        # TODO: REPLACE entire logic. if smask is not available it means the image is already opaque
        try:
            img = Image.open(io.BytesIO(img_bytes))            

            if img.mode != "RGBA":
                img = img.convert("RGBA")
            alpha = img.getchannel("A")
            has_transparent_bg = alpha.getextrema()[0] < 255  

            if has_transparent_bg:
                logging.debug("Transparent image : Adding white background....")

                # Add solid white background and use alpha as mask
                white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                white_bg.paste(img, (0,0), img)
                final_img = white_bg.convert("RGB")
            else:
                logging.debug("Opaque : Saving as it is.....")
                final_img = img.convert("RGB")

            return final_img
        
        except (IOError, OSError, ValueError, NameError) as e:
            logging.error(f"Error processing image: {e}")
            return None

    def get_elements(self) -> List[Dict]:
        all_elements = []

        for page_num, page in enumerate(self.doc):
            # Tables
            tables = page.find_tables()
            page_tables = []
            for tab in tables:
                df = tab.to_pandas()
                if df.empty:
                    continue

                page_tables.append({
                    "type": "table",
                    "content": "<Table here: Table's VLM generated description>",
                    "table_html": df.to_html(index=False),
                    "bbox": tab.bbox,
                    "page": page_num + 1,
                    "processed": False
                })

            # image + text and ignoring blocks inside table 
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])

            for block in blocks:
                block_rect = block["bbox"]
                is_inside_table=False
                for tab_obj in page_tables:
                    if self.__get_overlap_ratio(block_rect, tab_obj["bbox"]) > 0.6:
                        is_inside_table=True
                        
                        if not tab_obj["processed"]:
                            all_elements.append(tab_obj)
                            tab_obj["processed"]=True
                        break

                if is_inside_table:
                    continue
                
            # Images
            img_list = page.get_images(full=True)
            for img_item in img_list:
                bbox = page.get_image_bbox(img_item)
                xref = img_item[0]

                try:
                    ei = self.doc.extract_image(xref)
                except Exception as e:
                    logger.error(f"Error extracting image for xref={xref}: {e}")
                    continue

                raw_bytes = ei.get("image")
                smask_xref = ei.get("smask")


                # Calculate hash from raw extracted bytes
                img_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
                img_path = os.path.join(self.img_dir, f"{img_hash}.png")

                if os.path.exists(img_path):
                    logging.debug(f"Image already exists, skipping processing: {img_hash}.png")
                    continue

                img_bytes = self.__extract_and_apply_smask(raw_bytes, smask_xref)
                final_img = self.__process_image(img_bytes)
                if final_img is None:
                    logging.error(f"Failed to process image at page {page_num + 1}, skipping...")
                    continue

                final_img.save(img_path, format="PNG")
                logging.debug(f"Saved image: {img_hash}.png")

                all_elements.append({
                    "type": "image",
                    "content": "<Image here: VLM generated description>",
                    "image_path": img_path,
                    "bbox": tuple(bbox),
                    "page": page_num + 1
                })

            # Text blocks
            for block in blocks:
                text = " ".join(
                    span.get("text", "") 
                    for line in block.get("lines", []) 
                    for span in line.get("spans", [])
                ).strip()

                if len(text) < 3:
                    continue

                block_rect = block.get("bbox")
                all_elements.append({
                    "type": "text",
                    "content": text,
                    "bbox": block_rect,
                    "page": page_num + 1
                })

            # Remaining tables
            for tab in page_tables:
                if not tab.get("processed"):
                    all_elements.append(tab)

        return all_elements

    @staticmethod
    def _bbox_gap(bbox_a: Tuple, bbox_b: Tuple) -> float:
        """Euclidean edge-distance between two axis-aligned bounding boxes.
        Returns 0 when they overlap or touch."""
        ax0, ay0, ax1, ay1 = bbox_a
        bx0, by0, bx1, by1 = bbox_b
        hgap = max(0.0, max(bx0 - ax1, ax0 - bx1))
        vgap = max(0.0, max(by0 - ay1, ay0 - by1))
        return math.sqrt(hgap ** 2 + vgap ** 2)

    def get_image_context_sequential(
        self,
        elements: Optional[List[Dict]] = None,
        context_window: int = 2,
    ) -> List[Dict]:
        """For each image element, collect `context_window` non-image elements
        immediately before and after it in document order.
        Best for single-column PDFs."""
        if elements is None:
            elements = self.get_elements()

        results = []
        for idx, elem in enumerate(elements):
            if elem.get("type") != "image":
                continue

            before, i = [], idx - 1
            while i >= 0 and len(before) < context_window:
                if elements[i]["type"] != "image":
                    before.insert(0, elements[i])
                i -= 1

            after, i = [], idx + 1
            while i < len(elements) and len(after) < context_window:
                if elements[i]["type"] != "image":
                    after.append(elements[i])
                i += 1

            results.append({
                "mode": "sequential",
                "index": idx,
                "image": elem,
                "before": before,
                "after": after,
            })
        return results

    def get_image_context_spatial(
        self,
        elements: Optional[List[Dict]] = None,
        context_window: int = 2,
        max_distance: float = 300.0,
    ) -> List[Dict]:
        """For each image, find the `context_window * 2` nearest same-page
        text/table elements ranked by bounding-box distance.
        Best for multi-column layouts and floating images."""
        if elements is None:
            elements = self.get_elements()

        results = []
        for elem in elements:
            if elem.get("type") != "image":
                continue

            candidates = [
                e for e in elements
                if e["page"] == elem["page"] and e["type"] != "image"
            ]
            candidates.sort(key=lambda e: self._bbox_gap(elem["bbox"], e["bbox"]))

            nearby = [
                c for c in candidates
                if self._bbox_gap(elem["bbox"], c["bbox"]) <= max_distance
            ][:context_window * 2]

            results.append({
                "mode": "spatial",
                "image": elem,
                "nearby": nearby,
            })
        return results

    def get_image_context(
        self,
        method: Literal['sequential', 'spatial'] = 'sequential',
        context_window: int = 2,
        max_distance: float = 300.0,
        elements: Optional[List[Dict]] = None,
        use_vlm: bool = False
    ) -> Dict[str, str]:
        """High-level convenience: returns a flat {image_path: surrounding_text}
        mapping using the chosen context extraction method."""
        if elements is None:
            elements = self.get_elements()

        mappings = {}
        if method == 'sequential':
            results = self.get_image_context_sequential(elements, context_window)
            mappings = {
                r['image']['image_path']: "".join(
                    e['content'] for e in r['before'] + r['after']
                    if e['type'] == 'text'
                )
                for r in results
            }
        else:
            results = self.get_image_context_spatial(elements, context_window, max_distance)
            mappings =  {
                r['image']['image_path']: "".join(
                    e['content'] for e in r['nearby']
                    if e['type'] == 'text'
                )
                for r in results
            }

        if use_vlm:
            self.mapping_to_vlm(mappings)

        return mappings

    def mapping_to_vlm(self, mapping):
        keyword_mapping = {}

        for img, desc in mapping.items():
            # print(img)
            try:
                resp = ollama.generate(
                        model='ministral-3:3b',
                        prompt=f"""
Context: 
{desc}

--------------------------------

Extract keywords from the context that best match the image description as comma seperated values. if the image doesn\'t match the context, reply ONLY with 'None'""",
                        images=[img]
                ).response

                if resp == None or resp.strip().lower() == "none":
                    continue

                keyword_mapping.update({img.split('/')[-1]: resp})
            except KeyboardInterrupt:
                break

            except Exception:
                continue

        return keyword_mapping

    def close(self):
        """Close the PDF document."""
        self.doc.close()

if __name__=="__main__":
    pdfpre = PDFPreprocessor(r"data/Machine Learning.pdf")
    elems = pdfpre.get_image_context(method='spatial')
    print(f"{elems=}")
