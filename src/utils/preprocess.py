from PIL import Image
import io

def resize_image(image_path, max_size=(512, 512)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return buf.getvalue()