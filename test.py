import cv2 as cv
import numpy as np

img1 = cv.imread('horse.jpg')
img2 = cv.imread('image.orig/731.jpg')
sift = cv.SIFT_create()

g1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
g2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(g1, None)
kp2, des2 = sift.detectAndCompute(g2, None)

index_params = dict(algorithm = 1, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) >= 3:
    srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv.findHomography(srcPts, dstPts, cv.RANSAC, 5.0)

    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, H)

    cv.polylines(img2, [np.int32(dst)], True, (0, 0, 255))

result = cv.drawMatchesKnn(img1, kp1, img2, kp2, [good], None)
cv.imshow('result', result)
cv.waitKey()