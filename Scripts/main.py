from os import read, stat_result
from numpy.lib.function_base import average
import torch
from utils import plotLearning
from tensorflow.python.keras.backend import observe_object_name
from Agent import SACAgent
import numpy as np
import NFSEnv
import time
from dist1 import get_dist1
from takeaction import take_action
from resetloc import reset_map
import win32gui
import win32con
import win32api
import win32com.client
import easyocr
reader=easyocr.Reader(['en'])
env=NFSEnv.NFS_Env(reader=reader)
n_episodes=1000
scores=[]
episode_hist=[]
save=0
train=int(input('Is this a continuation: 0:No  1:Yes'))
hwndMain = win32gui.FindWindow(None,"Need for Speed™ Most Wanted")
print(hwndMain)
win32gui.ShowWindow(hwndMain,5)
shell = win32com.client.Dispatch("WScript.Shell")
shell.SendKeys('%')
win32gui.SetForegroundWindow(hwndMain)
win32gui.SetWindowPos(hwndMain,None,0,0,1024,800,0x0040)
agent=SACAgent(mem_size=600,batch_size=16,train=train)
for i in range(n_episodes):
    print('Training')
    done=False
    score=0
    dis,_,b_=get_dist1(reader)
    print(f"Distance: {dis}")
    if dis<0.2:
        reset_map()
    state1,map1=env.observe()
    state_1=torch.cat((state1,map1),0)
    while not done:
        action=agent.get_action(state_1).detach()
        actions=action.tolist()
        state2,map2,reward,done,bu=env.act(round(actions[0][0],1),round(actions[0][1],1))
        state_2=torch.cat((state2,map2),0)
        score+=reward
        agent.add_to_buffer(state_1,action.cpu().numpy(),reward,state_2,done)
        state1=state2
        agent.learn()
        print(f'Reward: {score}')
        dis1,_,b_=get_dist1(reader)
        if dis1<0.2:
            time.sleep(2)
            if bu==False:
                done=True
        if save%100==0:
            agent.save_all_checkpoints()
            agent.save_all_models()
        save+=1

    # episode_hist.append(agent.epsilon)
    scores.append(score)
    avg_score=np.mean(scores[max(0,i-100):i+1])
    print(f'episode: {i},  score: {score}, average_score: {avg_score}')

    
filename='NFSMW.png'
x=[i+1 for i in range(n_episodes)]
plotLearning(x,scores,episode_hist,filename)
