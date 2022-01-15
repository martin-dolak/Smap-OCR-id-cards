import pyocr
import pyocr.builders

from PIL import Image

# Important to install Tesseract OCR locally and place a path to the .exe file here
pyocr.tesseract.TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
USE_LANG = 'eng'


class ImageReader2:
    def __init__(self):
        self.tool = pyocr.get_available_tools()[0]

    def read_text(self, image: []):
        im_pil = Image.fromarray(image)
        txt = self.tool.image_to_string(
            im_pil,
            lang=USE_LANG,
            builder=pyocr.builders.TextBuilder()
        )

        return txt
