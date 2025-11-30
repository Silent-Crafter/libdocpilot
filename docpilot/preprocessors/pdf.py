import os
import fitz
import hashlib
from typing import List, Optional, Union, Literal, Dict, Tuple
import pprint

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
                    "content":html_text,
                    "bbox":tab.bbox,
                    "page":page_num+1,
                    "processed":False
                })
            
            #now images + text and ignoring blocks inside table
            page_dict=page.get_text("dict")
            blocks=page_dict["blocks"]

            for block in blocks:
                block_rect=block["bbox"]
                is_inside_table=False
                for tab_obj in page_tables:
                    if self._get_overlap_ratio(block_rect,tab_obj["bbox"])>0.6:
                        is_inside_table=True

                        if not tab_obj["processed"]:
                            all_elements.append(tab_obj)
                            tab_obj["processed"]=True
                        break

                if is_inside_table:
                    continue
                
                if(block["type"]==1):
                    img_bytes=block["image"]
                    ext=block["ext"]
                    img_hash=hashlib.sha256(img_bytes).hexdigest()
                    img_path=f"{self.img_dir}/{img_hash}.{ext}"

                    if not os.path.exists(img_path):
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                    
                    all_elements.append({
                        "type":"image",
                        "content":"<Image here: VLM generated description>",
                        "image_path":img_path,
                        "bbox":block["bbox"],
                        "page":page_num+1
                    })

                elif(block["type"]==0):
                    text=""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text+=span["text"]+" "
                    
                    if(len(text.strip())<3):
                        continue
                        
                    all_elements.append({
                        "type":"text",
                        "content":text.strip(),
                        "bbox":block["bbox"],
                        "page":page_num + 1
                    })

            for tab_obj in page_tables:
                if not tab_obj["processed"]:
                    all_elements.append(tab_obj)

        return all_elements

    def close(self):
        self.doc.close()

if __name__=="__main__":
    pdfpre=PDFPreprocessor("../../test_data/Attention.pdf")
    returned_list=pdfpre.get_elements()
    pprint.pprint([i for i in returned_list if i["page"] in [9]])