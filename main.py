import sys

from ocr import OCR
from model import Model
from processor import Processor

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print('Invalid number of arguments')
		print('python main.py [file_name.png]')
		sys.exit()

	file_name = sys.argv[1]
	ocr = OCR()
	m = Model()
	model = m.load()
	p = Processor()

	text = ocr.run(file_name)	
	word_seq_train = p.process(text)
	yhat = model.predict(word_seq_train)	
	if yhat > 0.5:
		predicted = "REAL"
	else:
		predicted = "FAKE"
	print(f'Predicted: {predicted}')