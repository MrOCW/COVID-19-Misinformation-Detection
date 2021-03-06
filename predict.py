from ocr import OCR
from model import Model
from utils import *
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='OCR')
	parser.add_argument('-i','--input',type=str,required=True,help='Path to the input image')
	parser.add_argument('--oem',type=int,default=3,help='0    Legacy engine only.\n1    Neural nets LSTM engine only.\n2    Legacy + LSTM engines.\n3    Default, based on what is available.')
	parser.add_argument('--psm',type=int,default=6,help='0    Orientation and script detection (OSD) only.\n1    Automatic page segmentation with OSD.\n2    Automatic page segmentation, but no OSD, or OCR.\n3    Fully automatic page segmentation, but no OSD. (Default)\n4    Assume a single column of text of variable sizes.\n5    Assume a single uniform block of vertically aligned text.\n6    Assume a single uniform block of text.\n7    Treat the image as a single text line.\n8    Treat the image as a single word.\n9    Treat the image as a single word in a circle.\n10    Treat the image as a single character.\n11    Sparse text. Find as much text as possible in no particular order.\n12    Sparse text with OSD.\n13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.')
	parser.add_argument('-w','--weights',type=str,help = 'Path to the weights')
	args = parser.parse_args()
	file_name = args.input
	oem = args.oem
	psm = args.psm
	ocr = OCR()
	m = Model()
	model = m.load(args.weights)
	text = ocr.run(file_name,oem,psm)
	print("----------TEXT TO CLASSIFY----------\n")
	print(text)
	print("------------------------------------")
	word_seq_train = process(text)
	yhat = model.predict(word_seq_train)	
	if yhat > 0.5:
		predicted = "Information seems OK!"
	else:
		predicted = "High Possibility of Misinformation!"
	print(f'Predicted: {predicted}')

