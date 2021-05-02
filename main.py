import cv2

from driver import Driver


def main():
    target_image_name = input("Enter the image name:\t")
    area_input = int(input("Enter minimum area of detectable object in the image: \t"))
    try:
        area: int = int(area_input)
        thresh_method_input = input("""Enter 0 for auto-thresh value (OTSU)
		or anything else for manual-thresh value (binary-thresh): \t""")
        if thresh_method_input == "0":
            thresh_value = 0
        else:
            thresh_value = int(input("\nEnter thresh value, ex: 220->\t"))
    except:
        print("Put legal values!")
        quit()

    image = cv2.imread(target_image_name)

    if image is None:
        print(
            "\nOops! Image assignment failed! \nCheck for typos and location of the image and also check thresh values")
        quit()
    else:
        test = Driver(image, area, thresh_method_input, thresh_value)
        test.detect()


if __name__ == "__main__":
    main()
