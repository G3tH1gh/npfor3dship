import torch
import gym
import pandas as pd
from DL.PPO import ppo_env1
import rl_utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import RLEnv

def train_on_policy_agent(env, agent, num_episodes, reply_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
            #    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
#               state, _ = env.Reset(seed=0)
                state = env.Reset()
                done = False
                while done is not True:
                    # state_copy = copy.deepcopy(state)
                    action = agent.take_action(state)
                    next_state, reward, done, _, info = env.step(action)
                    reply_buffer.add(state, action, reward, next_state, done)
                    # next_state_copy = copy.deepcopy(next_state)
                    state = next_state
                    episode_return += reward
                    if reply_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = reply_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                        '''
                        transition_dict['states'].append(state_copy)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state_copy)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        '''
                '''
                data = transition_dict['next_states']
                df = pd.DataFrame(data)
                df.to_excel(r"C:/Users\Magic秦\Desktop\state_lod.xlsx", index=False)
                '''
                return_list.append(episode_return)
            #    agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

minimal_size = 1000
batch_size = 64
buffer_size = 5000
actor_lr = 1e-3
critic_lr = 1e-4
num_episodes = 2000
hidden_dim = 256
gamma = 0.9
lmbda = 0.9
epochs = 10
epsilon = 0.2
device = torch.device('cuda:0') if torch.cuda.is_available() else RuntimeError("No GPU available!")

action_is_continuous = False
env = RLEnv.Env()
#env_name = 'MountainCar-v0'
#env = gym.make(env_name)
'''
state_dim = env.observation_space.shape[0]
if action_is_continuous:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n
'''
state_dim = 1
action_dim = 4
reply_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = ppo_env1.PPO(state_dim,action_dim,hidden_dim,actor_lr,critic_lr,lmbda,gamma,epsilon,epochs,device,action_is_continuous)

return_list = train_on_policy_agent(env,agent,num_episodes,reply_buffer,minimal_size,batch_size)

episodes_list = list(range(len(return_list)))
print(episodes_list)
print(return_list)

plt.plot(episodes_list,return_list)
plt.xlabel('episodes')
plt.ylabel('return')
plt.show()
mv_return = rl_utils.moving_average(return_list,11)
df = pd.DataFrame(mv_return, columns=['Max Q Value'])
df.to_excel(r"C:\Users\Magic秦\Desktop\reward.xlsx", index=False)
plt.plot(episodes_list,mv_return)
plt.xlabel('episodes')
plt.ylabel('return')
plt.show()

