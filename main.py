import cv2

from driver import Driver


def main():
	target_image_name = input("Enter the image name:\t")
	try:
		thesh_value_input = map(int, input("\nEnter Min and Max thresh value, ex: 40 255 ->\t").split())

		color_status = []
		for i in thesh_value_input:
			color_status.append(i)
	except:
		print("Put legal values!")
		quit()

	image = cv2.imread(target_image_name)

	if image is None and len(color_status) < 2:
		print(
			"\nOops! Image assignment failed! \nCheck for typos and location of the image and also check thresh values")
		quit()
	else:
		test = Driver(image, color_status)
		test.runDriver()

if __name__ == "__main__":
    main()