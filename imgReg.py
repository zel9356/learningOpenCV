import cv2
import numpy as np
import os
from PIL import Image, ImageOps

folder = "camImages\\"
cropFolder = "3croppedImages\\"


def crop():
    for f in os.listdir(folder):
        image1 = Image.open(folder + f)
        image2 = ImageOps.crop(image1, image1.size[1] // 3)
        image2.save(cropFolder + "c-" + f)


def newLoopedTest():
    pics = []
    for f in os.listdir(cropFolder):
        pics.append(f)
    for x in range(1, len(pics)):
        MAX_FEATURES = 500
        GOOD_MATCH_PERCENT = 0.05

        img1 = cv2.imread(cropFolder+ pics[0])
        img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1eq = cv2.equalizeHist(img1gray)

        img2 = cv2.imread(cropFolder + pics[x])
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2eq = cv2.equalizeHist(img2gray)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(img1eq, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2eq, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        name = (pics[x])
        # Draw top matches
        imMatches = cv2.drawMatches(img1eq, keypoints1, img2eq, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Use homography
        height, width, channels = img1.shape
        im2Reg = cv2.warpPerspective(img2, h, (width, height))
        name = ("3reg\\" + "reg-" + pics[x])
        cv2.imwrite(name, im2Reg)
        # cv2.imshow(name, im2Reg)
        print(name)
        # cv2.waitKey(0)


def main():
    crop()
    newLoopedTest()


if __name__ == '__main__':
    main()
