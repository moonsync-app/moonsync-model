from dataclasses import dataclass
import io, base64
from PIL import Image
from llama_index.readers.file.image import ImageReader
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
import uuid


@dataclass
class ImageHandler:
    image_url: str

    def run(self):
        print("IMAGE_URL", self.image_url[:100])
        id = str(uuid.uuid4())

        extension = self.image_url.split(",")[0].split("/")[1].split(";")[0]
        img = Image.open(
            io.BytesIO(base64.decodebytes(bytes(self.image_url.split(",")[1], "utf-8")))
        )
        img.save(f"/volumes/moonsync/data/img-{id}.{extension}")

        image_doc = ImageReader().load_data(
            file=f"/volumes/moonsync/data/img-{id}.{extension}"
        )

        openai_mm_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview", max_new_tokens=300
        )

        image_response = openai_mm_llm.complete(
            prompt="Describe the images as an alternative text. Give me a title and a detailed description for the image.",
            image_documents=image_doc,
        )

        return image_response
