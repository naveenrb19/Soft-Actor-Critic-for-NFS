from os import stat
from sys import path
from PIL.Image import new
import numpy as np
from numpy.lib.npyio import load
import torch.nn as nn
import torch
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.modules import padding
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.serialization import save
from ReplayBuffer import Replay_Buffer
from torch.jit._script import script
from Networks import Actor,Critic,ValueNetwork
from torch.autograd import Variable

class SACAgent():
    def __init__(self,mem_size,batch_size,train,gamma=0.99,alpha=0.2,tau=0.005,lera=3e-4):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lera=lera
        self.alpha=alpha
        self.gamma=gamma
        self.tau=tau
        self.train=train
        self.mem_size=mem_size
        self.batch_size=batch_size
        self.modelpath='D:/SAC_NFS/Models/'
        self.ckptpath='D:/SAC_NFS/Checkpoints/'
        self.policym='D:/SAC_NFS/Models1/policy.pth'
        self.q1m='D:/SAC_NFS/Models1/q1.pth'
        self.q2m='D:/SAC_NFS/Models1/q2.pth'
        self.valuem='D:/SAC_NFS/Models1/valuenet.pth'
        self.policyc='D:/SAC_NFS/Checkpoints1/policyck.pth'
        self.q1c='D:/SAC_NFS/Checkpoints1/q1ck.pth'
        self.q2c='D:/SAC_NFS/Checkpoints1/q2ck.pth'
        self.valuec='D:/SAC_NFS/Checkpoints1/valuenetck.pth'
        
        self.model_names=[self.modelpath+'policy.pth',self.modelpath+'q1.pth',self.modelpath+'q2.pth',self.modelpath+'value.pth']
        self.ckpt_names=[self.policyc,self.q1c,self.q2c,self.valuec]
        self.policynet=(Actor(6,16,2)).to(self.device)
        self.valuenet=(ValueNetwork(6,16)).to(self.device)
        self.targetvaluenet=(ValueNetwork(6,16)).to(self.device)
        self.Q1=(Critic(6,16,2)).to(self.device)
        self.Q2=(Critic(6,16,2)).to(self.device)
        self.q1_opt=optim.Adam(self.Q1.parameters(),lr=self.lera)
        self.q2_opt=optim.Adam(self.Q2.parameters(),lr=self.lera)
        self.value_opt=optim.Adam(self.valuenet.parameters(),lr=self.lera)
        self.policy_opt=optim.Adam(self.policynet.parameters(),lr=self.lera)
        self.optimizers=[self.policy_opt,self.q1_opt,self.q2_opt,self.value_opt]
        self.memory=Replay_Buffer(self.mem_size,self.batch_size)
        self.value_crtirerion=nn.MSELoss()
        self.Q1_criterion=nn.MSELoss()
        self.Q2_criterion=nn.MSELoss()
        if self.train==1:
            self.policynet=self.load_model(self.policym).to(self.device)
            self.Q1=self.load_model(self.q1m).to(self.device)
            self.Q2=self.load_model(self.q2m).to(self.device)
            self.valuenet=self.load_model(self.valuem).to(self.device)
            self.load_all_checkpoints()

        for target_param, param in zip(self.targetvaluenet.parameters(), self.valuenet.parameters()):
            target_param.data.copy_(param.data)
    def get_action(self,state):
        state=state.unsqueeze(0)
        mu,log_sigma=self.policynet(state.to(self.device))
        sigma=torch.exp(log_sigma)
        dist=Normal(mu,sigma)
        e=dist.rsample()
        action=torch.tanh(e)
        return action
    def add_to_buffer(self,state,action,reward,next_state,terminal):
        self.memory.store_transition(state,action,reward,next_state,terminal)
    def evaluate(self,state,epsilon=1e-6):
        batch_mu,batch_log_sigma=self.policynet(state.to(self.device))
        batch_sigma=torch.exp(batch_log_sigma)
        noise=Normal(0,1)
        z=noise.sample()
        action=torch.tanh(batch_mu+batch_sigma*z)
        log_prob=Normal(0,1).log_prob(batch_mu+batch_sigma*z) - torch.log(1-action.pow(2)+epsilon)
        return action,log_prob,z,batch_mu,batch_log_sigma
    def learn(self):
        state,action,reward,next_state,done=self.memory.sample_buffer()
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     =torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(done).unsqueeze(1).to(self.device)
    
        pred_q1_val=self.Q1(state,action)
        pred_q2_val=self.Q2(state,action)
        pred_value=self.valuenet(state)
        new_action,new_log_prob,z,new_mu,new_log_sigma=self.evaluate(state)
        # print(new_log_prob.shape)
        with torch.no_grad():
            target_value=self.targetvaluenet(next_state)
            target_q_value=reward+(1-done)*self.gamma*target_value
            # print(target_q_value.shape)
            q1_loss=self.Q1_criterion(pred_q1_val,target_q_value.detach())
            loss_q1 = Variable(q1_loss, requires_grad = True)
            q2_loss=self.Q2_criterion(pred_q2_val,target_q_value.detach())
            loss_q2 = Variable(q2_loss, requires_grad = True)
        self.q1_opt.zero_grad(set_to_none=True)
        loss_q1.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad(set_to_none=True)
        loss_q2.backward()
        self.q2_opt.step()
        with torch.no_grad():
            pred_new_q=torch.min(self.Q1(state,new_action),self.Q2(state,new_action))
            target_value=pred_new_q-self.alpha*(new_log_prob[:,0].unsqueeze(1)-new_log_prob[:,1].unsqueeze(1))
            # print(f'Value: {pred_value.shape}  Target: {target_value.shape}')
            loss_value=self.value_crtirerion(pred_value,target_value)
            value_loss= Variable(loss_value , requires_grad = True)
        self.value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_opt.step()
        with torch.no_grad():
            loss_policy=(new_log_prob-(self.alpha*pred_new_q)).mean()
            policy_loss = Variable(loss_policy, requires_grad = True)
        self.policy_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_opt.step()
        for target_param, param in zip(self.targetvaluenet.parameters(),self.valuenet.parameters()):
            target_param.data.copy_(
            target_param.data * (1.0 - self.tau) + param.data * self.tau)
    def save_checkpoint(self,model,opt,filename):
        print("----------------------------------------------------Saving checkpoint----------------------------------------------------")
        checkpoint={
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':opt.state_dict(),
                    }
        torch.save(checkpoint, filename)
    def load_model(self,path):
        return torch.load(path)
    def load_checkpoint(self,path, model, optimizer, lr):
        print("----------------------------------------------------Loading checkpoint---------------------------------------------------")
        checkpoint = torch.load(path,map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    def save_all_models(self):
        print('----------------------------------------------------Saving models--------------------------------------------------------')
        torch.save(self.policynet,self.policym)
        torch.save(self.Q1,self.q1m)
        torch.save(self.Q2,self.q2m)
        torch.save(self.valuenet,self.valuem)
    def save_all_checkpoints(self):
        self.save_checkpoint(self.policynet,self.policy_opt,self.policyc)
        self.save_checkpoint(self.Q1,self.q1_opt,self.q1c)
        self.save_checkpoint(self.Q2,self.q2_opt,self.q2c)
        self.save_checkpoint(self.valuenet,self.value_opt,self.valuec)
    def load_all_checkpoints(self):
        self.load_checkpoint(self.policyc,self.policynet,self.policy_opt,self.lera)
        self.load_checkpoint(self.q1c,self.Q1,self.q1_opt,self.lera)
        self.load_checkpoint(self.q2c,self.Q2,self.q2_opt,self.lera)
        self.load_checkpoint(self.valuec,self.valuenet,self.value_opt,self.lera)


