from pickle import NONE
import cv2 as cv
import numpy as np
from glob import glob

# the directory of the image database
database_dir = "image.orig"

# Compute pixel-by-pixel difference and return the sum
def compareImgs(img1, img2):
    # resize img2 to img1
	img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
	diff = cv.absdiff(img1, img2)
	return diff.sum()

def compareImgs_hist(img1, img2, channel):
	width, height = img1.shape[1], img1.shape[0]
	img2 = cv.resize(img2, (width, height))

	num_bins = 16
	hist1 = [0] * num_bins
	hist2 = [0] * num_bins
	bin_width = 255.0 / num_bins + 1e-4

	# compute histogram from scratch
	#for w in range(width):
	#	for h in range(height):
	# 		hist1[int(img1[h, w] / bin_width)] += 1
	# 		hist2[int(img2[h, w] / bin_width)] += 1

	# compute histogram by using opencv function
	# https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d

	hist1 = cv.calcHist([img1], [channel], None, [num_bins], [0, 255])
	hist2 = cv.calcHist([img2], [channel], None, [num_bins], [0, 255])
	
	# take average
	sum = 0
	for i in range(num_bins):
		sum += abs(hist1[i] - hist2[i])
	return sum / float(width * height)

def segmentation():
	img = cv.imread("beach.jpg")
#	img = cv.imread("image.orig/504.jpg")
	twoDimage = img.reshape((-1,3))
	twoDimage = np.float32(twoDimage)

	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 4
	attempts=1

	#Kmeans
	ret,label,center=cv.kmeans(twoDimage,K,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
	center = np.uint8(center)
	res = center[label.flatten()]
	result_image = res.reshape((img.shape))
#	return result_image 
	cv.imshow("Result", result_image)
	cv.waitKey(0)


def contourDetection(img):
#	img = cv.imread("bus.jpg")
#	img = cv.imread("image.orig/102.jpg")
#	divideImg(img, 5)

#	ret,thresh = cv.threshold(img, np.mean(img), 255, cv.THRESH_BINARY_INV)

#	edges = cv.dilate(cv.Canny(thresh,0,255),None)
#	edges = cv.erode(edges,None)
	test = cv.Canny(img,32,64)
	edges = cv.dilate(test,None)
	edges = cv.erode(edges,None)

	cnt = sorted(cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2], key=cv.contourArea)[-1]
	mask = np.zeros(img.shape[:2], dtype=np.uint8)
	masked = cv.drawContours(mask, [cnt],-1, 255, -1)

	dst = cv.bitwise_and(img, img, mask, mask)
	segmented = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
	return segmented

	cv.imshow("Result", dst)
	cv.waitKey(0)


def quantization(img, lowerbound, upperbound, type):
#	img = cv.imread("building.jpg")
	hsv_image_spot = cv.cvtColor(img, cv.COLOR_BGR2HSV)


	# Setting the black pixel mask and perform bitwise_and to get only the black pixels
	if(type == "Hue"):
		lowerbound = np.array([lowerbound,0,0])
		upperbound = np.array([upperbound,255,255])
	if (type == "Saturation"):
		lowerbound = np.array([0,lowerbound,0])
		upperbound = np.array([255,upperbound,255])
	if (type == "Vision"):
		lowerbound = np.array([0,0,lowerbound])
		upperbound = np.array([255,255,upperbound])
	mask = cv.inRange(hsv_image_spot, lowerbound, upperbound)
	masked = cv.bitwise_and(hsv_image_spot, hsv_image_spot, mask=mask)

	return masked

	cv.imshow("result", masked)
	cv.waitKey(0)

def thresholding():
	img = cv.imread("beach.jpg")
	img = cv.medianBlur(img,5)
#	segmentation(img)
	ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
	cv.imshow("Result", th1)
	cv.waitKey(0)

def calColorDistance(img1, img2):
	# convert image_A and image_B from BGR to LAB
#	img1 = cv.imread("image.orig/600.jpg").astype("float32")
#	img2 = cv.imread("image.orig/101.jpg").astype("float32")
	img2 = cv.resize(img2, (img1.shape[1], img1 .shape[0]))

	# compute difference
	diff = cv.add(img1,-img2)

	# separate into L,A,B channel diffs
	diff_L = diff[:,:,0]
	diff_A = diff[:,:,1]
	diff_B = diff[:,:,2]

	# compute delta_e as mean over every pixel using equation from  
	delta_e = np.mean( np.sqrt(diff_L*diff_L + diff_A*diff_A + diff_B*diff_B) )
	return delta_e
	print (delta_e)

def divideImg(img, num):
#	img = cv.imread("bus.jpg")
	(h, w) = img.shape[:2]
	(lcX, lcY) = (w // 3, h // 3)
	(rcX, rcY) = (w - w // 3, h - h // 3)

	area_1 = img[0:lcY, 0:lcX]
	area_2 = img[0:lcY, lcX:rcX]
	area_3 = img[0:lcY, rcX:w]

	area_4 = img[lcY:rcY, 0:lcX]
	area_5 = img[lcY:rcY, lcX:rcX]
	area_6 = img[lcY:rcY, rcX:w]

	area_7 = img[rcY:h, 0:lcX]
	area_8 = img[rcY:h, lcX:rcX]
	area_9 = img[rcY:h, rcX:w]

	(tcX,tcY) = (w // 6, h // 6)
	(bcX, bcY) = (w - w//6, h - h//6)
	centerMask = img[tcY : bcY, tcX : bcX]

	if(num == 1):
		return area_1
	if(num == 2):
		return area_2
	if(num == 3):
		return area_3
	if(num == 4):
		return area_4
	if(num == 5):
		return area_5
	if(num == 6):
		return area_6
	if(num == 7):
		return area_7
	if(num == 8):
		return area_8
	if(num == 9):
		return area_9
	if(num == 10):
		return centerMask


	return
	cv.imshow("Center", centerMask)
	cv.imshow("Top Left Corner", area_4)
	cv.imshow("Top Right Corner", area_5)
	cv.imshow("Bottom Right Corner", area_6)
	cv.waitKey(0)

def retrieval():
	print("1: beach")
	print("2: building")
	print("3: bus")
	print("4: dinosaur")
	print("5: flower")
	print("6: horse")
	print("7: man")
	choice = input("Type in the number to choose a category and type enter to confirm\n")
	if choice == '1':
		src_input = cv.imread("beach.jpg")
		print("You choose: %s - beach\n" % choice)
	if choice == '2':
		src_input = cv.imread("building.jpg")
		print("You choose: %s - building\n" % choice)
	if choice == '3':
		src_input = cv.imread("bus.jpg")
		print("You choose: %s - bus\n" % choice)
	if choice == '4':
		src_input = cv.imread("dinosaur.jpg")
		print("You choose: %s - dinosaur\n" % choice)
	if choice == '5':
		src_input = cv.imread("flower.jpg")
		print("You choose: %s - flower\n" % choice)
	if choice == '6':
		src_input = cv.imread("horse.jpg")
		print("You choose: %s - horse\n" % choice)
	if choice == '7':
		src_input = cv.imread("man.jpg")
		print("You choose: %s - man\n" % choice)	

	min_diff = 1e50
	max_diff = 0

	# src_input = cv.imread("man.jpg")

	cv.imshow("Input", src_input)

	# change the image to gray scale
	src_gray = cv.cvtColor(src_input, cv.COLOR_RGB2HSV)

	# read image database
	database = sorted(glob(database_dir + "/*.jpg"))


	for img in database:
		# read image
		img_rgb = cv.imread(img)

		# convert to gray scale
		img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)

		# compare the two images by abs diff
		absdiff = compareImgs(contourDetection(src_gray), contourDetection(img_gray)) / 10**8 / 4

#		test_src = contourDetection(divideImg(src_gray, 2))
#		test_img = contourDetection(divideImg(img_gray, 2))
#		absdiff = compareImgs(test_src, test_img) / 10**9
#		absdiff += compareImgs(contourDetection(divideImg(src_gray, 2)), contourDetection(divideImg(img_gray, 2))) / 10**9
#		absdiff += compareImgs(contourDetection(divideImg(src_gray, 3)), contourDetection(divideImg(img_gray, 3))) / 10**9
#		absdiff += compareImgs(contourDetection(divideImg(src_gray, 4)), contourDetection(divideImg(img_gray, 4))) / 10**9
#		absdiff += compareImgs(contourDetection(divideImg(src_gray, 5)), contourDetection(divideImg(img_gray, 5))) / 10**9
#		absdiff += compareImgs(contourDetection(divideImg(src_gray, 6)), contourDetection(divideImg(img_gray, 6))) / 10**9
#		absdiff += compareImgs(contourDetection(divideImg(src_gray, 7)), contourDetection(divideImg(img_gray, 7))) / 10**9
#		absdiff += compareImgs(contourDetection(divideImg(src_gray, 8)), contourDetection(divideImg(img_gray, 8))) / 10**9
#		absdiff += compareImgs(contourDetection(divideImg(src_gray, 9)), contourDetection(divideImg(img_gray, 9))) / 10**9

		# compare the two images by histogram, uncomment the following line to use histogram
		# H
#		histdiff = compareImgs_hist(quantization(src_gray, 1, 25, "Hue"), quantization(img_gray, 1, 25, "Hue"), 0) / 15
#		histdiff += compareImgs_hist(quantization(src_gray, 26, 40, "Hue"), quantization(img_gray, 26, 40, "Hue"), 0) / 15
#		histdiff += compareImgs_hist(quantization(src_gray, 41, 120, "Hue"), quantization(img_gray, 41, 120, "Hue"), 0) / 15
#		histdiff += compareImgs_hist(quantization(src_gray, 121, 190, "Hue"), quantization(img_gray, 121, 190, "Hue"), 0) / 15
#		histdiff += compareImgs_hist(quantization(src_gray, 191, 255, "Hue"), quantization(img_gray, 191, 255, "Hue"), 0) / 15
		# S
#		histdiff += compareImgs_hist(quantization(src_gray, 1, 51, "Saturation"), quantization(img_gray, 1, 51, "Saturation"), 1) / 15
#		histdiff += compareImgs_hist(quantization(src_gray, 52, 178, "Saturation"), quantization(img_gray, 52, 178, "Saturation"), 1) / 15
#		histdiff += compareImgs_hist(quantization(src_gray, 179, 255, "Saturation"), quantization(img_gray, 179, 255, "Saturation"), 1) / 15
		# V
#		histdiff += compareImgs_hist(quantization(src_gray, 52, 178, "Vision"), quantization(img_gray, 52, 178, "Vision"), 2) / 15
#		histdiff += compareImgs_hist(quantization(src_gray, 179, 255, "Vision"), quantization(img_gray, 179, 255, "Vision"), 2) / 15

#		condiff = calColorDistance(src_gray, img_gray)

		# compare by sift
#		diff = SIFT(contourDetection(src_gray), contourDetection(img_gray))

		# calMatchPoints
		diff = absdiff

		print(img, diff)

		# find the minimum difference
		if diff <= min_diff:
			# update the minimum difference
			min_diff = diff
			# update the most similar image
			closest_img = img_rgb
			result = img

#		if diff >= max_diff:
#			# update the minimum difference
#			max_diff = diff
#			# update the most similar image
#			closest_img = img_rgb
#			result = img
	
	print("the most similar image is %s, the pixel-by-pixel difference is %f " % (result, min_diff))
	print("\n")

	cv.imshow("Result", closest_img)
	cv.waitKey(0)
	cv.destroyAllWindows()



def SIFT(img1, img2):
	#img1 = cv.imread("flower.jpg")
	#img2 = cv.imread("image.orig/456.jpg")

	if img1 is None or img2 is None:
		print('Error loading images!')
		exit(0)
	#-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
	minHessian = 400
	detector = cv.SIFT_create()
	keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
	keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
	#-- Step 2: Matching descriptor vectors with a brute force matcher
#	matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)

	bf = cv.BFMatcher()
	matches = bf.knnMatch(descriptors1, descriptors2, k=2)

	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])

	return len(good) / len(matches)
#	matches = matcher.match(descriptors1, descriptors2)
	#-- Draw matches
#	img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
#	cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
	#-- Show detected matches
	#cv.imshow('Matches: SIFT (Python)', img_matches)
	#cv.waitKey()

	# draw good matches
#	matches = sorted(matches, key = lambda x:x.distance)
#	min_dist = matches[0].distance
#	good_matches = tuple(filter(lambda x:x.distance <= 2 * min_dist, matches))

	return matches[0].distance
	
	#-- Show detected matches
	img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
	cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	cv.imshow('Good Matches: SIFT (Python)', img_matches)
	cv.waitKey()


def main():
	# img = cv.imread("beach.jpg")
	# cv.imshow("Image", img)
	# from matplotlib import pyplot as plt
	# plt.hist(img.ravel(),10,[0,256]); plt.show()
	# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# cv.imshow("Gray Image", gray_img)
	# cv.waitKey()


	print("1: Image retrieval demo")
	print("2: SIFT demo")
	number = int(input("Type in the number to choose a demo and type enter to confirm\n"))
	if number == 1:
		retrieval()
	elif number == 2:
		divideImg()
		# pass
	else:
		print("Invalid input")
		exit()

main()