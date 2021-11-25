import mss
import cv2
import numpy as np
import matplotlib.pyplot as plt
def grab_state():
    with mss.mss() as sct:
        monitor = {"top": 20, "left": 0, "width": 1024, "height":768}
        img = np.array(sct.grab(monitor))
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return RGB_img
