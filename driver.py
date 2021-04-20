import cv2
# import argparse
import imutils

from shapeDetector import ShapeDetector


class Driver:
    def runDriver(self):
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-i", "--image", required=True,
        #     help="path to the input image")
        # args = vars(ap.parse_args())
        # image = cv2.imread(args["image"])
        image = cv2.imread("shapes.png")
        image = cv2.bitwise_not(image)

        resized = imutils.resize(image, width=300)
        ratio = image.shape[0] / float(resized.shape[0])
        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        sd = ShapeDetector()
        for c in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            shape = sd.detect(c)
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")

            cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (128, 128, 128), 2)
            # show the output image
            cv2.imshow("Image", image)
            cv2.waitKey(2000)
        cv2.destroyAllWindows()
