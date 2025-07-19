from mpe2 import simple_adversary_v3
import numpy as np
import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt

from sympy.codegen.ast import continue_

from maddpg_agent_v1 import Agent

start_time = time.time()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")


def multi_obs_to_state(multi_obs):
    state = np.array([])
    for agent_obs in multi_obs.values():
        state = np.concatenate([state, agent_obs])
    return state


NUM_EPISODE = 3
EPISODE = NUM_EPISODE
NUM_STEP = 50
MEMORY_SIZE = 10000
BATCH_SIZE = 512
TARGET_UPDATE_INTERVAL = 50
LR_ACTOR = 0.01
LR_CRITIC = 0.01
HIDDEN_DIM = 64
GAMMA = 0.99
TAU = 0.01

# 1 Initialize the agents
env = simple_adversary_v3.parallel_env(render_mode="human", N=2, max_cycles=NUM_STEP, continuous_actions=True,
                                       dynamic_rescaling=False)
multi_obs, infos = env.reset()
NUM_AGENT = env.num_agents
agent_name_list = env.agents

scenario = "simple_adversary_v3"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/models/" + scenario + "/"
timestamp = time.strftime("%Y%m%d%H%M%S")

# 1.1 Get obs_dim(可以和1.2的方式obs_dim = [env.observation_space(name).shape[0] for name in agent_name_list])
obs_dim = []
for agent_obs in multi_obs.values():
    obs_dim.append(agent_obs.shape[
                       0])  # 遍历multi_obs的值，并取出它的第一个维度是多少，若 agent_obs = [0.1, 0.5, -0.2, 1.0]，则 agent_obs.shape[0] 的值为 4
state_dim = sum(obs_dim)

# 1.2 Get action_dim
action_dim = []
for agent_name in agent_name_list:
    action_dim.append(env.action_space(agent_name).sample().shape[0])

agents = []
for agent_i in range(NUM_AGENT):
    print(f"Initializing agent {agent_i}...")
    agent = Agent(memo_size=MEMORY_SIZE, obs_dim=obs_dim[agent_i], state_dim=state_dim, n_agent=NUM_AGENT,
                  acion_dim=action_dim[agent_i], alpha=LR_ACTOR, beta=LR_CRITIC, fc1_dims=HIDDEN_DIM,
                  fc2_dims=HIDDEN_DIM, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE)  # TODO
    agents.append(agent)

episode_rewards = []
# 2 Main training loop
for episode_i in range(EPISODE):
    multi_obs, infos = env.reset()
    episode_reward = 0
    multi_done = {agent_name: False for agent_name in agent_name_list}
    for step_i in range(NUM_STEP):
        total_step = episode_i * NUM_STEP + step_i
        # 2.1 Collect actions from all agents
        multi_actions = {}
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_obs[agent_name]
            single_action = agent.get_action(single_obs)  # TODO
            multi_actions[agent_name] = single_action

        # 2.2 Execute action
        multi_next_obs, multi_reward, multi_done, multi_truncation, infos = env.step(multi_actions)
        state = multi_obs_to_state(multi_obs)
        next_state = multi_obs_to_state(multi_next_obs)

        if step_i >= NUM_STEP - 1:
            multi_done = {agent_name: True for agent_name in agent_name_list}

        # 2.3 Store memory  #每个agent都有自己的AC网络和buffer，当其中的agent不能工作，其它agent依旧正常
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_obs[agent_name]
            single_next_obs = multi_next_obs[agent_name]
            single_action = multi_actions[agent_name]
            single_reward = multi_reward[agent_name]
            single_done = multi_done[agent_name]
            agent.replay_buffer.add_memo(single_obs, single_next_obs, state, next_state,
                                         single_action, single_reward, single_done)  # TODO
            print(f"agent:single_action: {agent}:{single_action}")

        # 2.4 Update brain every fixed steps

        multi_batch_obses = []
        multi_batch_next_obses = []
        multi_batch_states = []
        multi_batch_next_states = []
        multi_batch_actions = []
        multi_batch_next_actions = []
        multi_batch_online_actions = []
        multi_batch_rewards = []
        multi_batch_dones = []

        # 2.4.1 Sample a batch of memories
        current_meno_size = min(MEMORY_SIZE, total_step + 1)
        if current_meno_size < BATCH_SIZE:
            batch_idx = range(0, current_meno_size)
        else:
            batch_idx = np.random.choice(current_meno_size, BATCH_SIZE)
        '''代码改进
        batch_size = min(current_memo_size, BATCH_SIZE)
        batch_idx = np.random.choice(current_memo_size, batch_size, replace=False)
        '''

        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            batch_obses, batch_next_obses, batch_states, batch_next_states, \
                batch_actions, batch_rewards, batch_dones = agent.replay_buffer.sample(batch_idx)  # TODO

            #  2.4.1.1 Single + batch
            batch_obses_tensor = torch.tensor(batch_obses, dtype=torch.float).to(device)
            batch_next_obses_tensor = torch.tensor(batch_next_obses, dtype=torch.float).to(device)
            batch_states_tensor = torch.tensor(batch_states, dtype=torch.float).to(device)
            batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float).to(device)
            batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.float).to(device)
            batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float).to(device)
            batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float).to(device)

            # 2.4.1.2 Multiple + batch
            multi_batch_obses.append(batch_obses_tensor)
            multi_batch_next_obses.append(batch_next_obses_tensor)
            multi_batch_states.append(batch_states_tensor)
            multi_batch_next_states.append(batch_next_states_tensor)
            multi_batch_actions.append(batch_actions_tensor)

            single_batch_next_action = agent.target_actor.forward(batch_next_obses_tensor)  # TODO
            multi_batch_next_actions.append(single_batch_next_action)

            single_batch_online_action = agent.actor.forward(batch_obses_tensor)  # TODO
            multi_batch_online_actions.append(single_batch_online_action)

            multi_batch_rewards.append(batch_rewards_tensor)
            multi_batch_dones.append(batch_dones_tensor)

        # 直接来自agent.replay_buffer.sample(batch_idx)，用于在线Critic网络actor前向传播求Q值，再用于最小化MSE的梯度优化
        multi_batch_actions_tensor = torch.cat(multi_batch_actions, dim=1).to(device)
        # 来自agent.target_actor.forward(batch_next_obses_tensor)，用于目标Critic'网络actor'前向传播求Q'值,求目标值y，再用于最小化MSE的梯度优化
        multi_batch_next_actions_tensor = torch.cat(multi_batch_next_actions, dim=1).to(device)
        # 来自agent.actor.forward(batch_obses_tensor)，用于更新Actor网络，
        multi_batch_online_actions_tensor = torch.cat(multi_batch_online_actions, dim=1).to(device)

        # 2.4.2 Update critic and actor
        #核心作用：通过间隔性更新目标网络，平衡训练稳定性与学习效率
        if (total_step + 1) % TARGET_UPDATE_INTERVAL == 0:

            for agent_i in range(NUM_AGENT):
                agent = agents[agent_i]

                batch_obses_tensor = multi_batch_obses[agent_i]
                batch_states_tentor = multi_batch_states[agent_i]
                batch_next_states_tentor = multi_batch_next_states[agent_i]
                batch_rewards_tentor = multi_batch_rewards[agent_i]
                batch_dones_tentor = multi_batch_dones[agent_i]
                batch_actions_tentor = multi_batch_actions[agent_i]

                # 2.4.2.1 Target critic
                critic_target_q = agent.target_critic.forward(batch_next_states_tentor,
                                                              multi_batch_next_actions_tensor.detach())  # TODO
                y = (batch_rewards_tentor + (1 - batch_dones_tentor) * agent.gamma * critic_target_q).flatten()

                # Critic
                critic_q = agent.critic.forward(batch_states_tentor,
                                                multi_batch_actions_tensor.detach()).flatten()  # TODO

                # Update critic
                critic_loss = nn.MSELoss()(y, critic_q)
                agent.critic.optimizer.zero_grad()
                critic_loss.backward()
                agent.critic.optimizer.step()

                # Update actor
                multi_batch_online_actions_list = [
                    single_batch_online_action if j == agent_i else single_batch_online_action.detach()
                    for j, single_batch_online_action in enumerate(multi_batch_online_actions)
                ]
                multi_batch_online_actions_tensor = torch.cat(multi_batch_online_actions_list, dim=1).to(device)
                actor_loss = agent.critic.forward(batch_states_tentor,
                                                  multi_batch_online_actions_tensor).flatten()  #这里其实为Q
                actor_loss = -torch.mean(actor_loss)    #为什么这么写？即对样本的Q值求平均。3)Actor更新（策略梯度）:通过Critic的指导（Q_{\mu}^i对动作的梯度），调整Actor策略μ_i以最大化长期奖励。
                agent.actor.optimizer.zero_grad()
                actor_loss.backward()
                agent.actor.optimizer.step()

                # Update target critic
                for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)
                    #与.parameters不同，对.data的操作不会影响梯度计算图，即不会被自动微分系统跟踪

                # Update target actor
                for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

        multi_obs = multi_next_obs
        episode_reward += sum([single_reward for single_reward in multi_reward.values()])
        print(f"Episode_reward:  {episode_reward}")

    # 3 Render the env
    if (episode_i + 1) % 100 == 0:
        env = simple_adversary_v3.parallel_env(N=2,
                                               max_cycles=NUM_STEP,
                                               continuous_actions=True,
                                               render_mode="human")

        for test_epi_i in range(5):
            multi_obs, infos = env.reset()
            for step_i in range(NUM_STEP):
                multi_actions = {}
                for agent_i, agent_name in enumerate(agent_name_list):
                    agent = agents[agent_i]
                    single_obs = multi_obs[agent_name]
                    single_action = agent.get_action(single_obs)  # TODO
                    multi_actions[agent_name] = single_action
                multi_next_obs, multi_reward, multi_done, multi_truncation, infos = env.step(multi_actions)
                multi_obs = multi_next_obs

    # 4 Save the agents' models
    if episode_i == 0:
        highest_reward = episode_reward
    if episode_reward > highest_reward:
        highest_reward = episode_reward
        print(f"Highest reward updated at episode {episode_i}: {round(highest_reward, 2)}")
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            flag = os.path.exists(agent_path)
            if not flag:
                os.makedirs(agent_path)
            torch.save(agent.actor.state_dict(), f"{agent_path}" + f"agent_{agent_i}_actor{scenario}_{timestamp}.path")
    print(f"episode_i: {episode_i}")
    episode_rewards.append(episode_reward)

end_time = time.time()
total_time = end_time - start_time
print(f'Total Duration: {total_time:.2f}')

plt.figure(figsize=(10, 5))
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.title('Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Review')
plt.grid(True)
plt.show()

env.close()
