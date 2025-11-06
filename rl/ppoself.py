import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim,action_is_continuous):
        super().__init__()
        if action_is_continuous:
            self.actor = nn.Sequential(
                nn.Linear(state_dim,hidden_dim),
                nn.ReLU(),  # max(0，x）
                nn.Linear(hidden_dim,action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.ReLU(),
                nn.Softmax(dim = -1)
            )

        self.critic = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
        )

    def forword(self):
        raise NotImplementedError

class PPO():
    def __init__(self,state_dim,action_dim,hidden_dim,lr_actor,lr_critic,lmbda,gamma,epsilon,epochs,device,action_is_continuous,entropy_coef=0.01):
        self.action_is_continuous = action_is_continuous
        self.ACnet = ActorCritic(state_dim,action_dim,hidden_dim,action_is_continuous).to(device)
        self.actor_optimizer = torch.optim.Adam(params=self.ACnet.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(params=self.ACnet.critic.parameters(), lr=lr_critic)
        self.lmbda = lmbda
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = device
        self.p = 0.3
        '''
        self.entropy_coef = entropy_coef

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        '''
    def take_action(self, state):
        if self.action_is_continuous:
            state = torch.tensor(state,dtype=torch.float).to(self.device)
            mu = 2 * self.ACnet.actor(state)
            action_dict = torch.distributions.Normal(mu,0.6)
            action = action_dict.sample()
            return action.item()

        else:
            # state = torch.tensor([state],dtype=torch.float).to(self.device)
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            probs = self.ACnet.actor(state)
            action = torch.argmax(probs).item() if torch.max(probs) > self.p else torch.distributions.Categorical(probs).sample().item()
            action = (action + 1)
        return action

    def update(self, transit_dict):
        states = torch.tensor(transit_dict['states'], dtype=torch.float).to(self.device)
       # actions = torch.tensor(transit_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transit_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transit_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transit_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)


        # rewards -= torch.mean(rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)


        td_target = rewards + self.gamma * self.ACnet.critic(next_states) * (1 - dones)
        td_delta = td_target - self.ACnet.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)


        if self.action_is_continuous:
            mu = self.ACnet.actor(states).detach()
            action_dists = torch.distributions.Normal(mu, 0.6)
            old_log_probs = action_dists.log_prob(actions)
        else:
            old_actor_params = copy.deepcopy(self.ACnet.actor.state_dict())
            self.ACnet.actor.load_state_dict(old_actor_params)
            old_log_probs = torch.log(self.ACnet.actor(states) + 1e-8).detach()
        #    print(self.ACnet.actor(states).detach())


        for _ in range(self.epochs):
            if self.action_is_continuous:
                mu = self.ACnet.actor(states)
                action_dists = torch.distributions.Normal(mu, 0.6)
                log_probs = action_dists.log_prob(actions)
            else:
                probs = self.ACnet.actor(states)
                log_probs = torch.log(probs + 1e-8)
               # entropy = -torch.sum(probs * log_probs, dim=-1).mean()  


            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage


            actor_loss = -torch.mean(torch.min(surr1, surr2))


            critic_value = self.ACnet.critic(states)
            critic_loss = F.mse_loss(critic_value, td_target.detach())


            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


            old_log_probs = log_probs.detach()
        #     for name, param in self.ACnet.named_parameters():
       #         if param.requires_grad:
        #            print(f"Parameter name: {name}, Gradient: {param.grad}")


     #   self.actor_scheduler.step()
      #  self.critic_scheduler.step()


