"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import cv2
import numpy as np
from ex1_utils import LOAD_GRAY_SCALE
from ex1_utils import imReadAndConvert



def gamma(value: int = 0):
    value = float(value) / 100

    new_img = np.array(image**value) # the range is 0-1 so we don't need to multi by 1
    cv2.imshow('Gamma Correction' , new_img)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    global image # for gammaDisplay
    image = imReadAndConvert(img_path , rep)

    # if is RGB we need change it to BGT for using cv2 functions
    if rep == 2:
        BGR = np.zeros_like(image)
        BGR[:,:,0] = image[:,:,2]
        BGR[:,:, 1] = image[:, :, 1]
        BGR[:,:,2] = image[:, :, 0]
        image = BGR

    # create GUI
    window = 'Gamma Correction'
    cv2.namedWindow('Gamma Correction')

    cv2.createTrackbar('gamma' , window , 0 , 200 , gamma)
    cv2.imshow(window , image)
    cv2.waitKey(0)

    cv2.destroyWindow(window)


def main():
    gammaDisplay(r'C:\Users\97253\Desktop\beach.jpg', LOAD_GRAY_SCALE)



if __name__ == '__main__':
     main()
