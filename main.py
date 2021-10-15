from ocr import OCR

if __name__ == "__main__":
	ocr = OCR()
	text = ocr.run()
	print(text)