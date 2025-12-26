import numpy as np
from PIL import Image
import io

IMG_SIZE = (180, 180)

def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 180, 180, 3)

    return img_array

