import cv2
# import argparse
import imutils

from shapeDetector import ShapeDetector


class Driver:
    def __init__(self, image, area, thresh_method_input, values):
        self.target_image = image
        self.min_area = area
        self.thresh_algo = thresh_method_input
        self.thresh_values = values

    def detect(self):

        resized = imutils.resize(self.target_image, width=300)
        ratio = self.target_image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale,
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # and threshold it
        if self.thresh_algo == "0":
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
        else:
            thresh = cv2.threshold(gray, self.thresh_values, 255, cv2.THRESH_BINARY)[1]

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
            if area > self.min_area:
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
            cv2.putText(self.target_image, result, (cX - 30, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (128, 254, 128), 2)
            cv2.putText(self.target_image, f'{area}p', (cX - 30, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 0, 128), 1)
            # show the output image
            cv2.imshow("Image", self.target_image)
            cv2.waitKey(2000)
        # end of for loop
        cv2.imshow("Image", self.target_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
