import base64
import os

from PIL import Image
from llama_index.core import Document
from io import BytesIO

from typing import List

def mappings_to_llamaindex_document(mapping: dict[str, str], image_dir: str, **kwargs) -> List[Document]:
    """
    Convert the image-label mapping into a llamaindex document that can be used to generate embeddings.
    The 'label' part will be embedded while the image will be in base64 format inside the metadata with the key
    'image'
    :param mapping: the mappings
    :param image_dir: The image directory
    :return: List of llamaindex documents
    """

    images = list(mapping.keys())
    labels = list(mapping.values())

    images = list(map(lambda i: os.path.abspath(os.path.join(image_dir, i)), images))

    docs: List[Document] = []
    for image, label in zip(images, labels):
        docs.append(
            Document(
                text=label,
                metadata={"file_name": image}
            )
        )

    return docs


def image_to_b64(img: str) -> str:
    image = Image.open(img)
    buffered = BytesIO()

    image.save(buffered, format="PNG")

    b64 = base64.b64encode(buffered.getvalue())
    b64 = b64.decode('utf-8')

    image.close()
    return b64
