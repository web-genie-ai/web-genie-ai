from PIL import Image
import io
import base64

from webgenie.constants import MAX_DEBUG_IMAGE_STRING_LENGTH


def pil_image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="jpeg")    
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    
    return base64_str


def image_to_base64(image_path: str) -> str:
    img = Image.open(image_path)
    return pil_image_to_base64(img)


def base64_to_image(base64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return img


def image_debug_str(base64_image: str) -> str:
    return base64_image[:MAX_DEBUG_IMAGE_STRING_LENGTH]
