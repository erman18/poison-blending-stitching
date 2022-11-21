
import cv2 as cv


def resizeImg(src, factor=0.3):
    #percent by which the image is resized
    if int(factor) == 1:
        return src

    # Compute the new dimensions of the image and resize it
    width = int(src.shape[1] * factor)
    height = int(src.shape[0] * factor)
    dsize = (width, height)
    return cv.resize(src, dsize)