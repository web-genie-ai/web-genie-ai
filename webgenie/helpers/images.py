from PIL import Image
import io
import base64

def pil_image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="jpeg")    
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    
    return base64_str

def image_to_base64(image_path: str) -> str:
    img = Image.open(image_path)
    return pil_image_to_base64(img)
