import os
import glob
import random
from PIL import Image
labels = ["CCCD", "HB", "XN_TTCT","SHK", "GSK", "Others"]
train = open('train.txt', 'w')
val = open('val.txt', 'w')

for fol in os.listdir("OCR_dataset"):
	try:
		total_files = glob.glob(f"OCR_dataset/{fol}/Original/**/*.jpg", recursive = True)
		idx = int(len(total_files)*0.9)
		if fol in labels: idx_label = labels.index(fol)
		else: idx_label = 5
		for i in total_files[:idx]:
			img = Image.open(i)
			train.write(f"{i}| {idx_label}"+'\n')
		for i in total_files[idx:]:
			img = Image.open(i)
			val.write(f"{i}| {idx_label}"+'\n')

	except:
		continue