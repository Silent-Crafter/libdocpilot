import os
from annotated_types import doc
import fitz
import hashlib
import io
from typing import List, Optional, Union, Literal, Dict, Tuple
import pprint
from PIL import Image


# from docpilot.notlogging.notlogger import NotALogger

# logger = NotALogger(__name__)
# logger.enabled = False

class PDFPreprocessor:

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

    def check_transparent_background(self, image: Image.Image) -> bool:
        if image.mode in ("RGBA", "LA"):
            alpha = image.getchannel("A")
            min_alpha = alpha.getextrema()[0]
            if min_alpha < 255:
                return True

        elif image.mode == "P":
            if "transparency" in image.info:
                return True
            try:
                rgba = image.convert("RGBA")
                return self.check_transparent_background(rgba)
            except:
                return False

        return False
        
    def extract_image_with_smask(self, doc, img_item) -> Tuple[bytes, str]:
        xref = img_item[0]
        smask_xref = img_item[1] if len(img_item) > 1 else 0

        try:
            # Extract the base color image
            ei = self.doc.extract_image(xref)
            raw_bytes = ei["image"]
            raw_ext = ei.get("ext", "png")

            if smask_xref > 0:
                print(f"smask={smask_xref} detected : applying manual alpha")

                # Load color image
                colour_img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")

                # Load SMask 
                smask_ei = self.doc.extract_image(smask_xref)
                mask_img = Image.open(io.BytesIO(smask_ei["image"])).convert("L")

                # Resize mask accordingly
                if mask_img.size != colour_img.size:
                    mask_img = mask_img.resize(colour_img.size, Image.LANCZOS)

                # Apply mask as alpha channel
                colour_img.putalpha(mask_img)

                # Convert back to bytes (PNG to preserve transparency)
                buf = io.BytesIO()
                colour_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                ext = "png"

            else:
                img_bytes = raw_bytes
                ext = raw_ext

            return img_bytes, ext

        except Exception as e:
            print(f"Warning: SMask handling failed for xref={xref}: {e}")
            try:
                pix = fitz.Pixmap(self.doc, xref)
                return pix.tobytes("png"), "png"
            except:
                base_image = self.doc.extract_image(xref)
                return base_image["image"], base_image.get("ext", "png")
        
    def process_image(self, img_bytes: bytes, ext: str, img_hash: str) -> str:
        try:
            img = Image.open(io.BytesIO(img_bytes))
            pil_format = (img.format or "").lower()

            if pil_format:
                original_ext = pil_format
            else: 
                original_ext = (ext or "png").lower().strip()

            if original_ext == "jpeg":
                original_ext = "jpg"
            

            if not original_ext.startswith('.'):
                original_ext = '.' + original_ext

            final_filename = f"{img_hash}{original_ext}"
            final_path = os.path.join(self.img_dir, final_filename)

            # check if image has transparent background and save accordingly
            has_transparent_bg = self.check_transparent_background(img)


            if has_transparent_bg:
                print(f"Transparent image : Adding white background : {final_filename}")

                if img.mode != "RGBA":
                    img = img.convert("RGBA")

                # Add solid white background and use alpha as mask
                white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                white_bg.paste(img, (0,0), img)
                final_img = white_bg.convert("RGB")

                if original_ext in [".jpg", ".jpeg"]:
                    final_img.save(final_path, "JPEG", quality=95, optimizer=True)
                else:
                    final_img.save(final_path, "PNG", optimize=True)

            else:
                print(f"Opaque : Saving as is : {final_filename}")
                    
                if original_ext in [".jpg", ".jpeg"]:
                    img = img.convert("RGB") if img.mode != "RGB" else img
                    img.save(final_path, "JPEG", quality=95)
                else:
                    img = img.convert("RGB") if img.mode in ("RGBA", "LA") else img
                    img.save(final_path, "PNG", optimize=True)

            return final_path
            
        except Exception as e:
            print(f"Error processing image: {e}")

    def get_elements(self) -> List[Dict]:
        all_elements=[]

        for page_num,page in enumerate(self.doc):
            #first finding tables
            tables=page.find_tables()
            table_bboxes=[fitz.Rect(tab.bbox) for tab in tables]

            page_tables=[]    

            for i, tab in enumerate(tables):
                df=tab.to_pandas()
                if df.empty: 
                    continue

                html_text=df.to_html(index=False)

                page_tables.append({
                    "type":"table",
                    "content":"<Table here: Table's VLM generated description>",
                    "table_html":html_text,
                    "bbox":tab.bbox,
                    "page":page_num+1,
                    "processed":False
                })
            
            #now images + text and ignoring blocks inside table
            img_list = page.get_images(full=True)
            for img_item in img_list:
                bbox = fitz.Rect(img_item[1:5])


                # Extract and process image
                img_bytes, ext = self.extract_image_with_smask(self.doc, img_item)
                img_hash = hashlib.sha256(img_bytes).hexdigest()[:16]
                print(f"Extracted image with hash {img_hash} and ext {ext} from page {page_num+1}")
                img_path = self.process_image(img_bytes, ext, img_hash)

                all_elements.append({
                    "type": "image",
                    "content": "<Image here: VLM generated description>",
                    "image_path": img_path,
                    "bbox": tuple(bbox),
                    "page": page_num + 1
                })

            # Text blocks
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])         

            for block in blocks:
                # Block must be a dictionary
                if not isinstance(block, dict):
                    continue

                if block.get("type") != 0:   
                    continue

                # Extract text safely
                text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span.get("text", "") + " "

                text = text.strip()
                if len(text) < 3:
                    continue

                # Skip text inside tables
                block_rect = block.get("bbox")
                if block_rect and any(self._get_overlap_ratio(block_rect, t["bbox"]) > 0.6 
                                    for t in page_tables):
                    continue

                all_elements.append({
                    "type": "text",
                    "content": text,
                    "bbox": block_rect,
                    "page": page_num + 1
                })

            # Append unprocessed tables
            for tab_obj in page_tables:
                if not tab_obj["processed"]:
                    all_elements.append(tab_obj)

        return all_elements

    def close(self):
        self.doc.close()

if __name__=="__main__":
    pdfpre=PDFPreprocessor("Data\\glob.pdf")
    returned_list=pdfpre.get_elements()
    pprint.pprint([i for i in returned_list if i["page"] in [3]])

    print("\n=== Extracted Images ===\n")
    for elem in returned_list:
        if elem.get("type") == "image":
            print(f"Page {elem['page']:2d} : {os.path.basename(elem['image_path'])}")

    pdfpre.close()