import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.stack(state),
                torch.tensor(action, dtype=torch.long, device=state[0].device),
                torch.tensor(reward, dtype=torch.float32, device=state[0].device),
                torch.stack(next_state),
                torch.tensor(done, dtype=torch.float32, device=state[0].device))
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, feature_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        phi = self.fc3(x)
        return phi

class BDQNAgent:
    def __init__(self, state_dim, action_dim, feature_dim, device, 
                 hidden_dim=64, lr=1e-3, gamma=0.99, buffer_capacity=10000, batch_size=64,
                 prior_var=1.0, noise_var=0.1, var_k=1.0, bayes_smoothing=0.9):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.q_network = QNetwork(state_dim, feature_dim, hidden_dim).to(device)
        self.target_network = QNetwork(state_dim, feature_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.var_k = var_k
        self.bayes_smoothing = bayes_smoothing
 
        self.ppt = [torch.zeros(feature_dim, feature_dim, device=device) for _ in range(action_dim)]
        self.py  = [torch.zeros(feature_dim, device=device) for _ in range(action_dim)]
        self.policy_mean = [torch.randn(feature_dim, device=device) * 0.01 for _ in range(action_dim)]
        self.policy_cov  = [torch.eye(feature_dim, device=device) for _ in range(action_dim)]
        self.sampled_mean = [m.clone() for m in self.policy_mean]
    
    def select_action(self, state):
        self.q_network.eval()
        with torch.no_grad():
            phi = self.q_network(state.unsqueeze(0)) 
            q_values = []
            for a in range(self.action_dim):
                q_val = torch.matmul(phi, self.sampled_mean[a]).item()
                q_values.append(q_val)
        self.q_network.train()
        return int(np.argmax(q_values))
    
    def update_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        phi = self.q_network(state_batch)
        with torch.no_grad():
            phi_next = self.target_network(next_state_batch)
        
        q_values = torch.zeros(self.batch_size, device=self.device)
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            a = action_batch[i].item()
            q_values[i] = torch.dot(phi[i], self.policy_mean[a])
            q_a_vals = torch.stack([torch.dot(phi_next[i], self.policy_mean[aa]) for aa in range(self.action_dim)])
            next_q_values[i] = torch.max(q_a_vals)
        
        target = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = torch.mean((q_values - target) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_bayesian_layer(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch_size = min(len(self.replay_buffer), self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)
        with torch.no_grad():
            phi = self.q_network(state_batch)
            phi_next = self.target_network(next_state_batch)
            q_next_vals = torch.zeros(batch_size, device=self.device)
            for i in range(batch_size):
                q_a_vals = torch.stack([torch.dot(phi_next[i], self.policy_mean[aa]) for aa in range(self.action_dim)])
                q_next_vals[i] = torch.max(q_a_vals)
            target_y = reward_batch + self.gamma * q_next_vals * (1 - done_batch)

        for i in range(batch_size):
            a = action_batch[i].item()
            phi_i = phi[i]
            outer = torch.ger(phi_i, phi_i)
            self.ppt[a] = self.bayes_smoothing * self.ppt[a] + (1 - self.bayes_smoothing) * outer
            self.py[a] = self.bayes_smoothing * self.py[a] + (1 - self.bayes_smoothing) * (phi_i * target_y[i])

        for a in range(self.action_dim):
            A = self.ppt[a] / self.noise_var + torch.eye(self.feature_dim, device=self.device) / self.prior_var
            try:
                invA = torch.inverse(A)
            except RuntimeError:
                invA = torch.pinverse(A)
            mu_a = torch.matmul(invA, self.py[a]) / self.noise_var
            sigma_a = self.var_k * invA
            self.policy_mean[a] = mu_a
            self.policy_cov[a] = sigma_a
            try:
                chol = torch.linalg.cholesky((sigma_a + sigma_a.t()) / 2.0)
            except RuntimeError:
                chol = torch.eye(self.feature_dim, device=self.device)
            epsilon = torch.randn(self.feature_dim, device=self.device)
            self.sampled_mean[a] = mu_a + torch.matmul(chol, epsilon)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filename):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'policy_mean': self.policy_mean,
            'policy_cov': self.policy_cov
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.policy_mean = checkpoint['policy_mean']
        self.policy_cov = checkpoint['policy_cov']
        self.update_target_network()

def train_bdqn(env, agent, num_episodes=100, bayes_update_freq=5, target_update_freq=10):
    total_rewards = []
    cumulative_returns = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, device=agent.device, dtype=torch.float32)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            agent.update_network()
        if ep % bayes_update_freq == 0:
            agent.update_bayesian_layer()
        if ep % target_update_freq == 0:
            agent.update_target_network()
        total_rewards.append(ep_reward)
        cumulative_return = (info['portfolio_value'] / env.initial_balance) - 1
        cumulative_returns.append(cumulative_return)
        print(f"Episode {ep+1}/{num_episodes}: Total Reward = {ep_reward:.4f}, Cumulative Return = {cumulative_return:.4f}")
    print("Training complete.")
    return total_rewards, cumulative_returns
