import cv2
import numpy as np
from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use("WebAgg")

print(cv2.__version__)

# Load an image
def cv01():
    win_name = 'win-01'
    cv2.namedWindow(win_name, 0)
    img = cv2.imread('img.png',1)
    img2 = cv2.imread('img.png',1)
    # img = cv2.resize(img, (100, 100))
    img2 = cv2.medianBlur(img2, 3)
    
    im_hc = cv2.hconcat([img, img2])

    cv2.imshow(win_name, im_hc)
    cv2.waitKey()

# cv01()


def cv02():
    img = cv2.imread('img.png', 1)
    img2 = cv2.imread('img.png', 1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (200, 200))
    plt.subplot(1, 2, 1) #řádek, sloupec, pozice
    plt.imshow(img)
    plt.subplot(1, 2, 2) #řádek, sloupec, pozice
    plt.imshow(img2)
    plt.show()

# cv02()

def cv03():
    win_name = 'win-01'
    cv2.namedWindow(win_name, 0)
    img = np.zeros([5, 5, 3], dtype=np.uint8) # 5x5 pixelů, 3 kanály

    img[1, 4] = [255, 255, 0]
    print(img[1, 4], img.shape)
    cv2.imshow(win_name, img)
    cv2.waitKey()

# cv03()
    
def cv04():
    FPS = 30.0
    video_size = (640, 480)
    cap = cv2.VideoCapture(0)
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS, video_size)

    while True:
        ret, frame = cap.read()
        if ret:
            video.write(frame)
            cv2.imshow('win_name', frame)
            if cv2.waitKey(1) == ord('q'):
                break

cv04()