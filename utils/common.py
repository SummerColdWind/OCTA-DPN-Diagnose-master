from PIL import Image

def load_image(path):
    image_raw = Image.open(path)
    return image_raw

