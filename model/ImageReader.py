import easyocr

USE_GPU = False
USE_LANG = ['cs']


class ImageReader:
    def __init__(self):
        self.reader = easyocr.Reader(USE_LANG, gpu=USE_GPU)

    def read_text(self, image):
        txt = self.reader.readtext(image, detail=0)

        return txt
