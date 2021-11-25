import win32gui
import win32con
import win32api
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
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
N = 0x31
LSHIFT=0x2A
SPACE=0x39
UP=0xc8    
LEFT=0xcb   
RIGHT=0xcd    
DOWN=0xd0
ESCAPE=0x01
ENTER=0x1c
WA=[0x11,0x1E]
WD=[0x11,0x20]
SA=[0x1F,0x1E]
SD=[0x1F,0x20]
a=[W,S,SPACE,WA,WD,SA,SD]
def take_action(acc,steer):
    if acc<0:
        t=abs(acc)-abs(steer)
        t=round(t,1)
        if steer>=0:
            PressKey(S)
            PressKey(D)
            time.sleep(abs(steer))
            ReleaseKey(D)
            try:
                time.sleep(t)
            except:
                pass
            ReleaseKey(S)
        else:
            PressKey(S)
            PressKey(A)
            time.sleep(abs(steer))
            ReleaseKey(A)
            try:
                time.sleep(t)
            except:
                pass
            ReleaseKey(S)
    if acc>0:
        t=abs(acc)-abs(steer)
        t=round(t,1)
        if steer>=0:
            PressKey(W)
            PressKey(D)
            time.sleep(abs(steer))
            ReleaseKey(D)
            try:
                time.sleep(t)
            except:
                pass
            ReleaseKey(W)
        else:
            PressKey(W)
            PressKey(A)
            time.sleep(abs(steer))
            ReleaseKey(A)
            try:
                time.sleep(t)
            except:
                pass
            ReleaseKey(W)