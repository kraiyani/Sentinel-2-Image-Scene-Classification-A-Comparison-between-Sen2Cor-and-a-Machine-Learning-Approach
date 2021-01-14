#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from django.contrib.gis.gdal import GDALRaster
import subprocess
import os
from joblib import dump, load
import sys
import getopt
from progress.bar import Bar
from collections import Counter
#import glob

def cal_index(b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b8a):

	brightness = wetness = avi = bsi = si = ndvi = ndwi = ndsi = ndgi = ndmi = nbri = npcri = ashburn = gvi = sci = 0

	brightness = 0.3037 * b2 + 0.2793 * b3 + 0.4743 * b4 + 0.5585 * b8 + 0.5082 * b10 + 0.1863 * b12

	brightness = np.round(brightness,4)

	wetness = 	0.1509 * b2 + 0.1973 * b3 + 0.3279 * b4 + 0.3406 * b8 - 0.7112 * b11 - 0.4572 * b12

	wetness = np.round(wetness,4)

	temp = (b8 * (1 - b4) * (b8 - b4))

	avi = np.cbrt(temp)
	avi = np.round(avi,4)

	temp = ((1 - b2) * (1 - b3) * (1 - b4))

	si = np.cbrt(temp)
	si = np.round(si,4)

	a = (b3 - b8)
	b = (b3 + b8)
	ndwi = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
	ndwi = np.round(ndwi,4)
	
	a = (b8 - b4)
	b = (b8 + b4)
	ndvi = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
	ndvi = np.round(ndvi,4)

	a = (b3 - b11)
	b = (b3 + b11)
	ndsi = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
	ndsi = np.round(ndsi,4)
	
	a = (b3 - b4)
	b = (b3 + b4)
	ndgi = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
	ndgi = np.round(ndgi,4)

	a = (b8 - b11)
	b = (b8 + b11)
	ndmi = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
	ndmi = np.round(ndmi,4)

	a = (b8 - b12)
	b = (b8 + b12)
	nbri = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
	nbri = np.round(nbri,4)

	a = (b4 - b2)
	b = (b4 + b2)
	npcri = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
	npcri = np.round(npcri,4)

	a = ((b11 + b4) - (b8 + b2))
	b = ((b11 + b4) + (b8 + b2))
	bsi = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
	bsi = np.round(bsi,4)

	ashburn = 2 * (b9 - b4)
	ashburn = np.round(ashburn,4)

	gvi = 0.7243 * b8 + 0.0840 * b11 - 0.1800 * b12 - 0.2848 * b2 - 0.2435 * b3 - 0.5436 * b4
	gvi = np.round(gvi,4)

	a = (b11 - b8)
	b = (b11 + b8)
	sci = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
	sci = np.round(sci,4)

	indexs = [brightness, wetness , avi , bsi , si , ndvi , ndwi , ndsi , ndgi , ndmi , nbri , npcri , ashburn , gvi , sci]

	return indexs

def Most_Common(lst):
	data = Counter(lst)
	value = data.most_common(1)[0][0]
	return value

def get_adjacent_indices(i, j, m, n, matrix):
	adjacent_values = []
	adjacent_indices = []

	adjacent_indices.append((i,j))
	# in middle
	if (i > 0) and (j > 0) and (i < m) and (j < n) and (i != m - 1) and (j != n - 1):
		# row above
		adjacent_indices.append((i-1,j-1))
		adjacent_indices.append((i-1,j))
		adjacent_indices.append((i-1,j+1))
		# same row
		adjacent_indices.append((i,j-1))
		adjacent_indices.append((i,j+1))
		# row below
		adjacent_indices.append((i+1,j-1))
		adjacent_indices.append((i+1,j))
		adjacent_indices.append((i+1,j+1))
	# four corners
	elif (i == 0) and (j == 0):
		# top left
		adjacent_indices.append((i,j+1))
		adjacent_indices.append((i+1,j))
		adjacent_indices.append((i+1,j+1))
	elif (i == m - 1) and (j == 0):
		# bottom left
		adjacent_indices.append((i-1,j))
		adjacent_indices.append((i-1,j+1))
		adjacent_indices.append((i,j+1))
	elif (i == m - 1) and (j == n - 1):
		# bottom right
		adjacent_indices.append((i,j-1))
		adjacent_indices.append((i-1,j-1))
		adjacent_indices.append((i-1,j))
	elif (i == 0) and (j == n - 1):
		# top right
		adjacent_indices.append((i,j-1))
		adjacent_indices.append((i+1,j-1))
		adjacent_indices.append((i+1,j))
	# top row no corner
	elif (i == 0) and (0 < j < n):
		# same row
		adjacent_indices.append((i,j-1))
		adjacent_indices.append((i,j+1))
		# row below
		adjacent_indices.append((i+1,j-1))
		adjacent_indices.append((i+1,j))
		adjacent_indices.append((i+1,j+1))
	# bottom row no corner
	elif (i == m - 1) and (0 < j < n):
		# same row
		adjacent_indices.append((i,j-1))
		adjacent_indices.append((i,j+1))
		# row above
		adjacent_indices.append((i-1,j-1))
		adjacent_indices.append((i-1,j))
		adjacent_indices.append((i-1,j+1))
	# first column no corner
	elif (j == 0) and (0 < i < m):
		adjacent_indices.append((i-1,j))
		adjacent_indices.append((i-1,j+1))
		adjacent_indices.append((i,j+1))
		adjacent_indices.append((i+1,j))
		adjacent_indices.append((i+1,j+1))
	# last column no corner
	elif (j == n - 1) and (0 < i < m):
		adjacent_indices.append((i-1,j))
		adjacent_indices.append((i-1,j-1))
		adjacent_indices.append((i,j-1))
		adjacent_indices.append((i+1,j-1))
		adjacent_indices.append((i+1,j))

	for x,y in adjacent_indices:
		adjacent_values.append(matrix[x][y])

	value = Most_Common(adjacent_values)
	if value == 0:
		return matrix[i][j]
	else:
		return value


def path_maker(path):
	if not os.path.exists(path):
		os.makedirs(path)

def error():
	print( 'L1C_to_six_class_creator.py -i <inputdirectory> -o <outputdirectory>')
	sys.exit()

def getRelevantDirectories(argv):
	inputDir = ''
	outputDir = ''
	modelDir = ''
	
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		error()

	for opt, arg in opts:
		if opt == '-h' or len(arg) == 0:
			error()
		if opt in ("-i", "--ifile"):
			inputDir = arg
		if opt in ("-o", "--ofile"):
			outputDir = arg

	if len(opts) == 0 or inputDir == '' or outputDir == '':
		print( 'L1C_to_six_class_creator.py -i <inputdirectory> -o <outputdirectory>')
		sys.exit()
	return inputDir, outputDir, modelDir


inputDir, outputDir, modelDir = getRelevantDirectories(sys.argv[1:])

file_name = inputDir + '.png'

modelDir = os.path.abspath(modelDir)
inputDir = os.path.abspath(inputDir)
outputDir = os.path.abspath(outputDir)

for root, dirs, files in os.walk(inputDir, topdown = False):
	for name in dirs:
		if name == 'IMG_DATA':
			inpath = os.path.abspath(os.path.join(root, name)) + '/'

model_path = os.path.join(modelDir,'model.joblib')

path_maker(outputDir)

# reading 13 bands
inpath = os.path.join(inpath,'*_B*.jp2')

# storing vrt file
outPath = os.path.join(outputDir,'resampled_stack.vrt')

# getting band value
command = "gdalbuildvrt -resolution user -tr 20 20 -separate -overwrite {0} {1}".format(outPath,inpath)
subprocess.run(command, shell=True)

rast = GDALRaster(outPath)
rastersBands = []

for band in rast.bands:
	b = band.data()
	b = np.true_divide(b,10000)
	rastersBands.append(b)

indexs = cal_index(rastersBands[0],rastersBands[1],rastersBands[2],rastersBands[3],rastersBands[4],rastersBands[5],rastersBands[6],rastersBands[7],
				rastersBands[8],rastersBands[9],rastersBands[10],rastersBands[11],rastersBands[12])

for index in indexs:
	rastersBands.append(index)

print('reading 13 bands + 12 indexs done')
rasterStack = np.dstack(rastersBands)

X_test = np.expand_dims(rasterStack,axis=0)

loaded_model = load(model_path)
rgbArray = np.zeros((X_test.shape[1],X_test.shape[2],3), 'uint8')
post_processingArray = np.zeros((X_test.shape[1],X_test.shape[2]),dtype='object')

bar = Bar('Classifying Image', max = X_test.shape[1])

def color(row,col,r,g,b):
	rgbArray[row,col, 0] = r * 255
	rgbArray[row,col, 1] = g * 255
	rgbArray[row,col, 2] = b * 255

ci = cl = ot = sh = sn = wa = 0

for row in range(X_test.shape[1]):
	in_arr = X_test[0][row]
	y_pred = loaded_model.predict(in_arr)
	for class_v,col in zip (y_pred,range(in_arr.shape[0])):
		post_processingArray[row][col] = class_v
		if class_v == 'cirrus':
			ci = ci + 1
			color(row,col,0.733, 0.773, 0.925) 
		elif class_v == 'cloud':
			cl = cl + 1
			color(row,col,0.949, 0.949, 0.949) 
		elif class_v == 'other':
			ot = ot + 1
			color(row,col,0, 1, 0) 
		elif class_v == 'shadow':
			sh = sh + 1
			color(row,col,0.467, 0.298, 0.043) 
		elif class_v == 'snow':
			sn = sn + 1
			color(row,col,0.325, 1, 0.980) 
		elif class_v == 'water':
			wa = wa + 1
			color(row,col,0, 0, 1) 
	bar.next()

total = ci + cl + ot + sh + sn + wa

print("\nci-{0}, cl-{1}, ot-{2}, sh-{3}, sn-{4}, wa-{5}, total-{6}\n".format(ci,cl,ot,sh,sn,wa,total))

img = Image.fromarray(rgbArray,'RGB')
# storing RGB file
file_name = file_name.split('/')[-1]
RGB_file_path = os.path.join(outputDir,file_name)
img.save(RGB_file_path)

print('image saved at',RGB_file_path)

command = "rm {0}".format(outPath)
subprocess.run(command, shell=True)

bar.finish()

bar = Bar('Post Processing', max = X_test.shape[1])

rgbArray = np.zeros((X_test.shape[1],X_test.shape[2],3), 'uint8')
m = post_processingArray.shape[0]
n = post_processingArray.shape[1]

ci = cl = ot = sh = sn = wa = 0

for row in range(m):
	for col in range(n):
		class_v = get_adjacent_indices(row,col,m,n,post_processingArray)
		if class_v == 'cirrus':
			ci = ci + 1
			color(row,col,0.733, 0.773, 0.925) 
		elif class_v == 'cloud':
			cl = cl + 1
			color(row,col,0.949, 0.949, 0.949) 
		elif class_v == 'other':
			ot = ot + 1
			color(row,col,0, 1, 0) 
		elif class_v == 'shadow':
			sh = sh + 1
			color(row,col,0.467, 0.298, 0.043) 
		elif class_v == 'snow':
			sn = sn + 1
			color(row,col,0.325, 1, 0.980) 
		elif class_v == 'water':
			wa = wa + 1
			color(row,col,0, 0, 1)
	bar.next()

total = ci + cl + ot + sh + sn + wa

print("\nci-{0}, cl-{1}, ot-{2}, sh-{3}, sn-{4}, wa-{5}, total-{6}\n".format(ci,cl,ot,sh,sn,wa,total))

img = Image.fromarray(rgbArray,'RGB')
# storing RGB file
file_name = 'post_processed_' + file_name.split('/')[-1]
RGB_file_path = os.path.join(outputDir,file_name)
img.save(RGB_file_path)

bar.finish()

print('image saved at',RGB_file_path)