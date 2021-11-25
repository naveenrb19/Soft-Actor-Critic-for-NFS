from gameover import game_over
from directkeys import *
from resetloc import reset_map
import win32gui
import win32con
import win32api
from directkeys import *
from time import sleep
import random
import cv2
from observestate1 import grabstatet1
from grabstatetp1 import grabstatetp1 
from takeaction import take_action
from dist1 import get_dist1
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
import easyocr
class NFS_Env(object):
    def __init__(self,reader):
        self.reader=reader
        self.reward=0
        self.speed1=0
        self.speed2=0
        self.speed3=0
        self.indis=0
        self.dist1=0
        self.mdiff2=0
        self.dist2=0
        self.indisf,self.odo1,_=get_dist1(self.reader)
        self.odo2=self.odo1
        self.skip=0
    
    def observe(self):
        img,map_img,self.speed1=grabstatet1()
        
        return img,map_img
    def act(self,acc,steer):
        self.reward=0
        chk=0
        take_action(acc,steer)
        time.sleep(1)
        self.dist2,self.odo2,bust=get_dist1(self.reader)
        img,map_img,self.speed2,crash=grabstatetp1()
        mdiff1=self.indis-self.dist2
        odiff=self.odo2-self.odo1
        print(f'S1: {self.speed3}       D1: {self.dist1}        O1: {self.odo1}     Map Diff: {mdiff1}')
        print(f'S2: {self.speed2}       D2: {self.dist2}        O2: {self.odo2}     Odo Diff: {odiff}')
        print(f'Busted:  {bust}')
        if self.skip==1:
            if self.speed3!=None and self.speed2!=None:
                if int(self.speed2-self.speed3)<=4 and int(self.speed2-self.speed3)>0:
                    self.reward+=1
                    chk=1
                    if self.speed2<180:
                        self.speed3=self.speed2
                    
                    # if odiff>0.01: 
                    #     self.reward=self.reward+2
                    if mdiff1>self.mdiff2:
                        self.reward=self.reward+(5*abs(mdiff1-self.mdiff2))
                    elif mdiff1<self.mdiff2:
                        self.reward=self.reward-(3*abs(mdiff1-self.mdiff2))
                    else:
                        self.reward=self.reward
                    
            
                else:
                    if chk==0:
                        self.reward=0
                        # if odiff>0.01: 
                        #     self.reward=self.reward+1 
                        if mdiff1>self.mdiff2:
                            self.reward=self.reward+(2*abs(mdiff1-self.mdiff2))
                        elif mdiff1<self.mdiff2:
                            self.reward=self.reward-(3*abs(mdiff1-self.mdiff2))
            
            if bust==True:
                self.reward=self.reward-15
                time.sleep(5)
            if crash==True:
                self.reward=self.reward-10
                time.sleep(4)
        self.mdiff2=mdiff1
        self.skip=1   
        # elif self.speed2<self.speed1:
        #     self.reward=-1
        # elif self.rew:
        #     self.reward=0
        # if mdiff<-0.01:
        #     self.reward=50
        # elif mdiff>0.01:
        #     self.reward=-3
        # else:
        #     self.reward=0
        
        # if self.odo2>self.odo1:
        #     self.reward=2
        # elif self.odo1==self.odo2:
        #     self.reward=-1
        game=game_over()
        
        self.odo1=self.odo2
        return img,map_img,self.reward,game,bust
    def reset(self):
        reset_map()




        