import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sympy.stats.rv import probability
from torch.distributions.normal import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class ReplayMemory:
    def __init__(self, memo_capacity, state_dim, action_dim):
        self.memo_size = memo_capacity
        self.state_memo = np.zeros((self.memo_size, state_dim))
        self.next_state_memo = np.zeros((self.memo_size, state_dim))
        self.action_memo = np.zeros((self.memo_size, action_dim))
        self.reward_memo = np.zeros(self.memo_size)
        self.done_memo = np.zeros(self.memo_size)
        self.memo_counter = 0

    def add_memo(self, state, action, reward, next_state, done):
        index = self.memo_counter % self.memo_size
        self.state_memo[index] = state
        self.next_state_memo[index] = next_state
        self.action_memo[index] = action
        self.reward_memo[index] = reward
        self.done_memo[index] = done

        self.memo_counter += 1

    def sample_memory(self, batch_size):
        current_momo_size = min(self.memo_counter, self.memo_size)
        batch = np.random.choice(current_momo_size, batch_size, replace=False)
        batch_state = self.state_memo[batch]
        batch_action = self.action_memo[batch]
        batch_reward = self.reward_memo[batch]
        batch_next_state = self.next_state_memo[batch]
        batch_done = self.done_memo[batch]

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.q = nn.Linear(self.fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], 1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q


class ValueNetwork(nn.Module):
    def __init__(self, beta, state_dim, fc1_dim, fc2_dim):  # ValueNetwork中没有action_dim
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        self.fc1 = nn.Linear(self.state_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.v = nn.Linear(self.fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)

        return v


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.max_action = max_action

        self.fc1 = nn.Linear(self.state_dim, self.fc1_dim)  # a = pi(s)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)

        self.mu = nn.Linear(self.fc2_dim, self.action_dim)
        self.sigma = nn.Linear(self.fc2_dim, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.tiny_positive = 1e-6

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu(x)) * self.max_action  # a ~ [-2,2]

        sigma = self.sigma(x)
        sigma = F.softplus(sigma) + self.tiny_positive
        sigma = torch.clamp(sigma, min=self.tiny_positive, max=1.0)

        return mu, sigma

    def sample_normal(self, state, reparameterize):
        mu, sigma = self.forward(state)
        probability = Normal(mu, sigma)

        if reparameterize:
            raw_action = probability.rsample()  # a = mu + sigma * epsilon
        else:
            raw_action = probability.sample()

        tanh_action = torch.tanh(raw_action)  # [-inf, inf] --> [-1, 1]
        scaled_action = tanh_action * self.max_action
        log_prob = probability.log_prob(raw_action)  # log mu 原始动作在高斯分布下的对数概率
        log_prob -= torch.log(1 - tanh_action.pow(2) + self.tiny_positive)  # 修正项：补偿tanh变换导致的概率密度变化（雅可比行列式修正）

        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(0)
        log_prob = log_prob.sum(1, keepdim=True)

        return scaled_action, log_prob


class SAC_agent:
    def __init__(self, state_dim, action_dim, memo_capacity,
                 alpha, beta, gamma, tau,
                 layer1_dim, layer2_dim, batch_size):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayMemory(memo_capacity=memo_capacity, state_dim=state_dim, action_dim=action_dim)
        self.action_dim = action_dim
        self.critic_1 = CriticNetwork(beta=beta, state_dim=state_dim,
                                      action_dim=action_dim, fc1_dim=layer1_dim,
                                      fc2_dim=layer2_dim).to(device)
        self.critic_2 = CriticNetwork(beta=beta, state_dim=state_dim,
                                      action_dim=action_dim, fc1_dim=layer1_dim,
                                      fc2_dim=layer2_dim).to(device)
        self.value = ValueNetwork(beta=beta, state_dim=state_dim,
                                  fc1_dim=layer1_dim, fc2_dim=layer2_dim).to(device)
        self.target_value = ValueNetwork(beta=beta, state_dim=state_dim,
                                         fc1_dim=layer1_dim, fc2_dim=layer2_dim).to(device)
        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim, fc1_dim=layer1_dim,
                                  fc2_dim=layer2_dim, max_action=2).to(device)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        action, _ = self.actor.sample_normal(state, reparameterize=False)
        return action.cpu().detach().numpy()

    def add_memo(self, state, action, reward, next_state, done):
        self.memory.add_memo(state, action, reward, next_state, done)

    def update(self):
        if self.memory.memo_counter < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_memory(self.batch_size)

        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.bool).to(device)

        value = self.value(state).view(-1)

        with torch.no_grad():
            value_ = self.target_value(new_state).view(-1)
            value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward()
        self.value.optimizer.step()

        # Update target value network
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Actor network
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value  # Eq. (12)
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Soft Q-function
        with torch.no_grad():
            q_hat = reward + self.gamma * value_

        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)

        critic_1_loss = F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = F.mse_loss(q2_old_policy, q_hat)

        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2.optimizer.step()
