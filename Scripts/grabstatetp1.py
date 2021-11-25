import time
import cv2
import mss
import numpy
import pytesseract
import PIL
from re import search
import torch
from torchvision.transforms import transforms
transform_norm = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize(0,1)])
start_point_crash = (15,90)
end_point_crash = (220, 170) 
st_pt_map=(0,540)
end_pt_map=(250,768)
speed_st=(900,690)
speed_end=(980,740)
# Blue color in BGR 
color = (255, 0, 0) 
thickness = 2
speed_mem=[0]
len_b=0
len_a=0
i=0
reward=0
crshtxt='CRASHED'
kernel1 =numpy.array([[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1]])
pytesseract.pytesseract.tesseract_cmd = r'E:\Ques\Tesseract-OCR\tesseract.exe'

def grabstatetp1():
    
    with mss.mss() as sct:
    
        monitor = {"top": 20, "left": 0, "width": 1024, "height":768}

    
            
            
            
        img = numpy.array(sct.grab(monitor))
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        map_img=RGB_img[540:768,0:250]
        speed=RGB_img[690:690+50,875:900+65,0]
        crashed=img[90:170,15:300]
        img_b=speed
        img_b[img_b<210]=0
        img_b[img_b>210]=255
        edge_speed= cv2.Canny(img_b,222, 230)
        dil=cv2.dilate(edge_speed,kernel1)
#         gray = cv2.cvtColor(speed, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_b,200,300, cv2.THRESH_BINARY)
#         otsu_thresh_image = PIL.Image.fromarray(thresh)
        dil2=cv2.dilate(thresh,kernel1)
        spd=pytesseract.image_to_string(dil2, lang="letsgodigital", config="--psm 8 -c tessedit_char_whitelist=0123456789")
        crashed=pytesseract.image_to_string(crashed)
        try:
            spd=''.join(c for c in spd if c.isdigit())
            spd=int(eval(spd))
        except:
            spd=None
    RGB_img=cv2.resize(RGB_img,(640,640), interpolation = cv2.INTER_AREA)
    map_img=cv2.resize(map_img,(640,640), interpolation = cv2.INTER_AREA)
    if search('CRASHED',crashed) or search('TAKEN DOWN',crashed):
        return transform_norm(RGB_img),transform_norm(map_img),spd,True
    else:
        return transform_norm(RGB_img),transform_norm(map_img),spd,False  
      

        

       