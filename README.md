# Soft-Actor-Critic-for-NFS
**Soft actor critic implementation on NFS environment. An attempt to bring autonomous driving in NFS environment using soft actor critic model. (Ongoing)**

<!--------------------------------->
## Architecture
<p align='justify'>
The architecture includes an Actor network to output which action to take.
A Critic network to evaluate state action pairs.
A value network to evaluate value of being in a state.
NFS env class handles operations such as getting input image,executing actions and tracking values such as map distance,odometer distance and speed in order to assign reward.
  
**Networks in the script**
  
  * Policy Network (Adam optimizer)
  
  * Q1 value Network (Adam optimizer)
  
  * Q2 value Network (Adam optimizer)
  
  * Value Network (Adam optimizer)
  
  * Targetvalue Network
</p>

<!------------------------------->
## Flow of code
<p align='justify'>
 In the main.py code, initial distance is checked if found zero code opens the map and randomly moves to a point and sets the destination. Next image (640x640x3) of entire game screen is taken and concatenated with image of map alone (resized to 640x640x3) and sent to Actor network. 

The Actor network outputs mean and standard deviation(clamped between -1,1). Using the mean and standard deviation an action(size=1x2) is sampled using tanh function. Out of two actions first is used for acceleratioor deceleration and second for steer right or left. Sign of the output value determines the direction while magnitude determines length of key press. After taking action,(state,action,reward,nextstate,done) are stored in replay buffer. 
  
In the learning loop, using target value network target value is calculated using the formula:
  
**_Target Q Value=Reward + (1-Done) x Discount Factor x Calculated Target value using targetvalue network_**  
  
Using Q value networks Q1 and Q2 two values are obtained for a state action pair.
Networks Q1 and Q2 are trained using Mean Squared Error loss using Q value and Target Q value pairs.

**_Q1 loss= MSE(Predicted Q1 value, Target Q value)_**

**_Q2 loss= MSE(Predicted Q2 value, Target Q value)_**

Next loss value for value network is obtained and value network is trained. First states are sent through actor network and new actions are obtained.
 
  **_Predicted Value=valuenet(states)_**
  
  **_Predicted new Q value = min(Q1(state,new action),Q2(state,new action))_**
  
  **_Target Value = Predicted new Q value - alpha * logprob(new actions)_**
  
  **_Value loss=MSE(Predicted value,Target Value)_**
 
Next up we calculate the policy loss as follows:
  
  **_Policy loss = Mean(logprob(new actions)-alpha x predicted new Q value)_**
  
  After updating all 4 networks targetvalue networ parameters are soft updated with value network parameters.
  
  Note: I have not trained it for long time to visualize the result. Model too seems to be not converging. I would like to make this one successful. Ive worked on this for many days and I need collaborations or new perspectives from people more established in this domain. It would be really helpful.
 </p>
