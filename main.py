import sys
from ocr import OCR

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print('Invalid number of arguments')
		print('python main.py [file_name.png]')
		sys.exit()

	file_name = sys.argv[1]
	ocr = OCR()
	text = ocr.run(file_name)
	print(text)