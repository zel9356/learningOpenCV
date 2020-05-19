import cv2
from matplotlib import pyplot as plt
import numpy as np


def displayWithCV2():
    # creating an "image" variable
    # reading in an image
    image1 = cv2.imread("images\C3C23.tiff")
    image2 = cv2.imread("images\IMG8208.jpg", cv2.IMREAD_GRAYSCALE)  # make grey scale, default is IMREAD_COLOR

    # displaying image
    # gibe the display a title and then an img to display
    cv2.imshow('Image1', image1)
    cv2.imshow('Image2', image2)
    # wait till any key is pressed and then close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def displayWithMatplotLib():
    # we are reading in out colored imag as a grey scale image
    image2 = cv2.imread("images\IMG8208.jpg", cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread("images\C3C23.tiff")
    # image, color map: gray
    plt.imshow(image2, cmap='gray')
    # the image will display with tck marks
    plt.show()
    # once we click out of the first image the second one will show
    plt.imshow(image1)
    plt.xticks([]), plt.yticks([])  # empty list: plotting no tick marks
    plt.show()
    img = cv2.imread("images\OC14.tiff")
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    # using matplotlib we can esily plot lines on our image
    plt.plot([200, 300, 400], [100, 200, 300], 'c', linewidth=5)
    plt.show()


def openVideoCamera():
    # using first webcam on computer (0), for second it would be 1 and so on
    cap = cv2.VideoCapture(0)

    # forever
    """This code initiates an infinite loop (to be broken later by a break statement),
     where we have ret and frame being defined as the cap.read(). Basically, 
     ret is a boolean regarding whether or not there was a return at all, 
     at the frame is each frame that is returned. If there is no frame, 
     you wont get an error, you will get None.
    """
    while (True):
        ret, frame = cap.read()
        # converts frame to grey
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # display that grey frame
        frame = cv2.flip(frame, 1)  # 1 flips horizontally, 0 vertically, -1 both
        cv2.imshow('frame', frame)
        # if q is pressed break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def captureFromCam():
    vid = cv2.VideoCapture(0)
    img_counter = 0;
    while (True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        if not ret:
            print("failed to grab frame")
            break
        frame = cv2.flip(frame, 1)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            # string of name we will give image
            img_name = "opencv_frame_{}.png".format(img_counter)
            # save image frame as img_name
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    vid.release()
    cv2.destroyAllWindows()


def threshold():
    # We are reading in one of the images of tail
    img1 = cv2.imread("images\C3C23.tiff")
    """
    "Here, the matter is straight-forward. For every pixel,
    the same threshold value is applied. If the pixel value is 
    smaller than the threshold, it is set to 0, otherwise it is set 
    to a maximum value. The function cv.threshold is used to apply the
    thresholding. The first argument is the source image, which should
    be a grayscale image. The second argument is the threshold value which 
    is used to classify the pixel values. The third argument is the maximum
    value which is assigned to pixel values exceeding the threshold. OpenCV 
    provides different types of thresholding which is given by the fourth
    parameter of the function. Basic thresholding as described above is done 
    by using the type cv.THRESH_BINARY."
    https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    """
    # pixels smaller than 125 are black, higher white
    retval, threshold = cv2.threshold(img1, 125, 255, cv2.THRESH_BINARY)
    # resizing so we can see enitre image (h and w divied by 4)
    img = cv2.resize(img1, (612, 512))
    cv2.imshow('original', img)
    # resizing threshold image
    img = cv2.resize(threshold, (612, 512))
    cv2.imshow('threshold', img)
    img1 = cv2.imread("images\C3C23.tiff", cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    th = cv2.resize(th, (612, 512))
    cv2.imshow('Adaptive threshold', th)
    retval2, threshold2 = cv2.threshold(img1, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = cv2.resize(threshold2, (612, 512))
    cv2.imshow("Otsu", th2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edgeDetection():
    """
    First argument is our input image. Second and third arguments are our minVal and
    axVal respectively. Third argument is aperture_size. It is the size of Sobel
    kernel used for find image gradients. By default it is 3. Last argument is
    2gradient which specifies the equation for finding gradient magnitude. If it is
    True, it uses the equation mentioned above which is more accurate, otherwise it
    uses this function: Edge_Gradient(G)=|Gx|+|Gy|. By default, it is False.
    """
    image1 = cv2.imread("images\OC14.tiff")
    edges = cv2.Canny(image1, 100, 200)
    image1 = cv2.resize(image1, (612, 512))
    edges = cv2.resize(edges, (612, 512))
    cv2.imshow("Original", image1)
    cv2.imshow("Edge detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gradients():
    frame = cv2.imread("images\OC14.tiff")
    """https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html
    Sobel operators is a joint Gausssian smoothing plus differentiation operation, 
    so it is more resistant to noise. You can specify the direction of derivatives to 
    be taken, vertical or horizontal (by the arguments, yorder and xorder respectively). 
    You can also specify the size of kernel by the argument ksize. If ksize = -1, a 3x3 
    Scharr filter is used which gives better results than 3x3 Sobel filter. 
    Please see the docs for kernels used.
    It calculates the Laplacian of the image given by the relation, Δsrc=∂2src∂x2+∂2src∂y2 
    where each derivative is found using Sobel derivatives. 
    If ksize = 1, then following kernel is used for filtering:kernel=⎡⎣⎢0101−41010⎤⎦⎥
    """

    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    frame = cv2.resize(frame, (612, 512))
    laplacian = cv2.resize(laplacian, (612, 512))
    sobelx = cv2.resize(sobelx, (612, 512))
    sobely = cv2.resize(sobely, (612, 512))
    cv2.imshow('Original', frame)
    # cv2.imshow('Mask', mask)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('sobelx', sobelx)
    cv2.imshow('sobely', sobely)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    openVideoCamera()


if __name__ == '__main__':
    main()
