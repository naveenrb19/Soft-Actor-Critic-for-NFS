import pytesseract as pt
from grabstate import grab_state
pt.pytesseract.tesseract_cmd = r'E:\Ques\Tesseract-OCR\tesseract.exe'
import cv2
import numpy as np
def game_over():
    kernel1 =np.array([[-1,-1]])
    img_b=grab_state()[75:110,20:73]
    img_b[img_b<140]=0
    img_b[img_b>140]=255
    edge_speed= cv2.Canny(img_b,222, 230)
    dil=cv2.dilate(img_b,kernel1)
    ret, thresh = cv2.threshold(dil,125,220, cv2.THRESH_BINARY)
    dil2=cv2.dilate(thresh,kernel1)
    blurImg = cv2.blur(dil2,(3,1))
    spd=pt.image_to_string(blurImg, lang="letsgodigital", config="--psm 8 -c tessedit_char_whitelist=.0123456789")
    if len(spd)==1:
        return True
    else:
        return False
