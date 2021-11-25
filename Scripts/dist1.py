import time
import cv2
import mss
import numpy
import pytesseract
import PIL
from re import search
import torch
from torchvision import transforms
transform_norm = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize(0,1)])
md=0
od=0
def get_dist1(reader):
    global md
    global od
    with mss.mss() as sct:
    
        monitor = {"top":20, "left": 0, "width": 1024, "height":768}
        img = numpy.array(sct.grab(monitor))
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        odo_dist=RGB_img[735:768,900:975,0]
        map_dist=RGB_img[75:110,20:80,0]
        busted=RGB_img[260:300,120:255,0]
    bound1=reader.readtext(map_dist)
    try: 
        d=bound1[0][1]
        d=''.join(c for c in d if c.isdigit())
    #     print(d)
        d=eval(d)
        md=d
    except:
        d=md
    
    bound2=reader.readtext(odo_dist)
    try: 
        o=bound2[0][1]
        o=''.join(c for c in o if c.isdigit())
    #     print(d)
        o=eval(o)
        od=o
    except:
        o=od
    bound3=reader.readtext(busted)
    try:
        bust=bound3[0][1]
        bust=''.join(c for c in bust)
    except:
        bust='Not busted'
    if search('BUSTED',bust):
        return d,o,True
    else:
        return d,o,False
