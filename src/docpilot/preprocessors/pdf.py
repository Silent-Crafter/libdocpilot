import io
import pprint
import hashlib
import os
from typing import List, Optional, Union, Dict, Tuple
import fitz
from annotated_types import doc
from PIL import Image


# from docpilot.notlogging.notlogger import NotALogger

# logger = NotALogger(__name__)
# logger.enabled = False

class PDFPreprocessor:
    """A preprocessor for extracting elements from PDF documents.
    
    This class handles extraction of text, images, and tables from PDF files,
    with support for image processing including soft masks and transparency handling.
    """

    def __init__(self, file_path: Optional[Union[str, os.PathLike[str]]]) -> None:
        if not isinstance(file_path, (str, os.PathLike)):
            raise TypeError(f"Expected str, os.PathLike got {type(file_path).__name__}")

        self.file_path=file_path
        self.doc=fitz.open(file_path)
        self.img_dir="out_images/"
        os.makedirs(self.img_dir, exist_ok=True)

    def _get_overlap_ratio(self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
        x0=max(bbox1[0], bbox2[0])
        y0=max(bbox1[1], bbox2[1])
        x1=min(bbox1[2], bbox2[2])
        y1=min(bbox1[3], bbox2[3])

        if(x0>=x1 or y0>=y1):
            return 0.0
        
        inter_area=(x1 - x0) * (y1 - y0)
        bbox1_area=(bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

        if(bbox1_area==0):
            return 0.0        
        return inter_area/bbox1_area
        
    def extract_image_with_smask(self, img_item) -> Tuple[bytes, str] | None:
        """Extract image with soft mask (SMask) support from PDF.
        
        Args:
            img_item: Image item tuple containing xref and optional smask_xref.
            
        Returns:
            1) Tuple of (image_bytes, image_format) where image_bytes is the processed image
            and image_format is the file extension (e.g., 'png').
            2) None if extraction fails.
        """
        xref = img_item[0]
        smask_xref = img_item[1] if len(img_item) > 1 else 0

        try:
            ei = self.doc.extract_image(xref)
        except Exception as e:
            print(f"Error extracting image for xref={xref}: {e}")
            return None
        
        raw_bytes = ei["image"]

        if smask_xref > 0:
            print(f"smask={smask_xref} detected : applying manual alpha")

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
            return img_bytes, "png"          
             
        return raw_bytes, ei.get("ext", "png")
            
    def process_image(self, img_bytes: bytes) -> str | None:
        """Process an image by converting to RGBA, handling transparency, and saving.
        
        Args:
            img_bytes: Raw image bytes to process.
            
        Returns:
            1) Path to the saved processed image.
            2) None if processing fails.
        """    
        try:
            img = Image.open(io.BytesIO(img_bytes))            

            if img.mode != "RGBA":
                img = img.convert("RGBA")
            alpha = img.getchannel("A")
            has_transparent_bg = alpha.getextrema()[0] < 255  

            if has_transparent_bg:
                print("Transparent image : Adding white background....")

                # Add solid white background and use alpha as mask
                white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                white_bg.paste(img, (0,0), img)
                final_img = white_bg.convert("RGB")
            else:
                print("Opaque : Saving as it is.....")                    
                final_img = img.convert("RGB")

            return final_img
        
        except (IOError, OSError, ValueError, NameError) as e:
            print(f"Error processing image: {e}")
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
                    if self._get_overlap_ratio(block_rect, tab_obj["bbox"]) > 0.6:
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
                bbox = fitz.Rect(img_item[1:5])

                result = self.extract_image_with_smask(img_item)
                if result is None:
                    print(f"Failed to extract image at page {page_num + 1}, skipping...")
                    continue

                img_bytes, _ = result

                final_img = self.process_image(img_bytes)
                if final_img is None:
                    print(f"Failed to process image at page {page_num + 1}, skipping...")
                    continue

                temp_path = os.path.join(self.img_dir, "temp.png")
                final_img.save(temp_path, format="PNG")

                with open(temp_path, "rb") as f:
                    img_bytes = f.read()

                # Calculate hash from raw extracted bytes
                img_hash = hashlib.sha256(img_bytes).hexdigest()[:16]
                img_path = os.path.join(self.img_dir, f"{img_hash}.png")

                if os.path.exists(img_path):
                    print(f"Image already exists, skipping processing: {img_hash}.png")
                    os.remove(temp_path)  # Clean up temp file
                else:
                    final_img.save(img_path, format="PNG")
                    print(f"Saved image: {img_hash}.png")

                all_elements.append({
                    "type": "image",
                    "content": "<Image here: VLM generated description>",
                    "image_path": img_path,
                    "bbox": tuple(bbox),
                    "page": page_num + 1
                })

            # Text blocks
            for block in blocks:
                # if not isinstance(block, dict) or block.get("type") != 0:
                #     continue

                text = " ".join(
                    span.get("text", "") 
                    for line in block.get("lines", []) 
                    for span in line.get("spans", [])
                ).strip()

                if len(text) < 3:
                    continue

                block_rect = block.get("bbox")
                # if block_rect and any(self._get_overlap_ratio(block_rect, t["bbox"]) > 0.6 
                #                     for t in page_tables):
                #     continue

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

    def close(self):
        """Close the PDF document."""
        self.doc.close()

if __name__=="__main__":
    pdfpre=PDFPreprocessor("Data\\Machine Learning.pdf")
    returned_list = pdfpre.get_elements()
    pprint.pprint([i for i in returned_list if i["page"] in [3]])

    print("\n=== Extracted Images ===\n")
    for elem in returned_list:
        if elem.get("type") == "image":
            print(f"Page {elem['page']:2d} : {os.path.basename(elem['image_path'])}")
    pdfpre.close()
