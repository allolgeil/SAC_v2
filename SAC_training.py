import os
import time
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


from SAC_agent import SAC_agent

scenario = 'Pendulum-v1'
env = gym.make(scenario)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
MEMORY_SIZE = 100000

agent = SAC_agent( state_dim=STATE_DIM, action_dim=ACTION_DIM, memo_capacity=MEMORY_SIZE,
                 alpha=3e-4, beta=3e-4, gamma=0.99, tau=0.005,
                 layer1_dim=64, layer2_dim=64, batch_size=256)  # TODO

NUM_EPISODE = 100
NUM_STEPS = 200

# The directory to save model
current_path = os.path.dirname(os.path.realpath(__file__))
model_dir = current_path + '/models/'
os.makedirs(model_dir, exist_ok=True)
timestamp = time.strftime('%Y%m%d%H%M%S')

REWARD_BUFFER = []
best_reward = env.reward_range[0]
PLOT_REWARD = True
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0.0
    for step_i in range(NUM_STEPS):
        action = agent.get_action(state)  # TODO
        next_state, reward, done, trunc, info = env.step(action)
        # 错next_state, reward, terminated, truncated, info = env.step(action)
        agent.add_memo(state, action, reward, next_state, done)  # TODO
        episode_reward += reward
        state = next_state
        agent.update()  # TODO
        if done:
            break
    REWARD_BUFFER.append(episode_reward)
    avg_reward = np.mean(REWARD_BUFFER)

    # Save model
    if avg_reward > best_reward:
        best_reward = avg_reward
        torch.save(agent.actor.state_dict(), model_dir + f'sac_actor_{timestamp}.pth')
        print(f'...saving model with best reward: {best_reward}')

    print(f'Episode {episode_i:.1f}, avg_reward {avg_reward:.1f}')
    #print(f'Episode {episode_i}', 'reward %.1f' %episode_reward, 'avg_reward %.1f' %avg_reward)


env.close()

if PLOT_REWARD:
    plt.plot(np.arange(len(REWARD_BUFFER)), REWARD_BUFFER, color='purple', alpha=0.5, label='Reward')
    plt.plot(np.arange(len(REWARD_BUFFER)), gaussian_filter1d(REWARD_BUFFER, sigma=5), color='purple', linewidth=2)
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig(f'Reward-{scenario}-{timestamp}.png', format='png')
    plt.show()