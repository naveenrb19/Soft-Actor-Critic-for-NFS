import numpy as np
from tensorflow.python.keras.backend import dtype
class Replay_Buffer(object):
    def __init__(self,mem_size,batch_size):
        self.mem_size=mem_size
        self.state_size=(self.mem_size,6,640,640)
        self.state_mem=np.zeros(self.state_size,dtype=np.uint8)
        self.new_state_mem=np.zeros(self.state_size,dtype=np.uint8)
        self.action_mem=np.zeros((self.mem_size,2),dtype=np.int8)
        self.reward_mem=np.zeros(self.mem_size)
        self.terminal_state_mem=np.zeros(self.mem_size)
        self.batch_size=batch_size
        self.mem_cntr=0
    def store_transition(self,state,action,reward,new_state,terminal_state):
        index=self.mem_cntr%self.mem_size
        self.state_mem[index]=state
        self.action_mem[index]=action
        self.reward_mem[index]=reward
        self.new_state_mem[index]=new_state
        self.terminal_state_mem[index]=1-(terminal_state)
        self.mem_cntr=self.mem_cntr+1
        
    def sample_buffer(self):
        max_mem=min(self.mem_cntr,self.batch_size)
        batch=np.random.choice(max_mem,self.batch_size)
        state=self.state_mem[batch]
        action=self.action_mem[batch]
        reward=self.reward_mem[batch]
        new_state=self.new_state_mem[batch]
        terminal_state=self.terminal_state_mem[batch]
        return state,action,reward,new_state,terminal_state







    