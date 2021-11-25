from directkeys import *
from time import sleep
import random
import cv2
UP=0xc8    
LEFT=0xcb   
RIGHT=0xcd    
DOWN=0xd0
ESCAPE=0x01
ENTER=0x1c
TAB=0x0f
a=[UP,DOWN,LEFT,RIGHT]
def reset_map():
    PressKey(TAB)
    sleep(3)
    ReleaseKey(TAB)
    for i in range(3):
        k=random.choice(a)
        PressKey(k)
        sleep(random.randint(1,3))
        ReleaseKey(k)
    sleep(2)
    PressKey(ENTER)
    sleep(0.6)
    ReleaseKey(ENTER)
    sleep(1)
    PressKey(ENTER)
    sleep(0.6)
    ReleaseKey(ENTER)
    sleep(3)
    PressKey(ESCAPE)
    sleep(1)
    ReleaseKey(ESCAPE)
