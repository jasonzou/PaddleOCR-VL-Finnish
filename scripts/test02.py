import paddleocr
print(paddleocr.__version__)

# Try to list available configurations
from paddleocr import PaddleOCR
help(PaddleOCR.__init__)
