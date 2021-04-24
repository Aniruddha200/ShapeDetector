import cv2
# import argparse
import imutils

from shapeDetector import ShapeDetector


class Driver:
    def __init__(self, image, thesh_values):
        self.target_image = image
        self.target_image_inverted = cv2.bitwise_not(self.target_image)
        self.thesh_values = thesh_values

    def runDriver(self):

        resized = imutils.resize(self.target_image_inverted, width=300)
        ratio = self.target_image.shape[0] / float(resized.shape[0])
        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, self.thesh_values[0], self.thesh_values[1], cv2.THRESH_BINARY)[1]

        cv2.imshow("amu", thresh)

        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        sd = ShapeDetector()
        for c in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            area = cv2.contourArea(c)
            if area > 500:
                result = sd.detect(c)
            else:
                continue

            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)

            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(self.target_image, [c], -1, (0, 127, 254), 1)
            cv2.putText(self.target_image, result, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (221, 254, 128), 2)
            cv2.putText(self.target_image, f'{area}', (cX, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (127, 128, 128), 2)
            # show the output image
        cv2.imshow("Image", self.target_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
