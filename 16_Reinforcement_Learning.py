# # TensorFlow Tutorial #16

# # Reinforcement Learning (Q-Learning)

# ## Introduction

# ## The Problem

# ## Q-Learning

# ### Simple Example

# ### Detailed Example

# ## Motion Trace

# ## Training Stability

# ## Flowchart

# ## Neural Network Architecture

# ## Installation

# ## Imports  # In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time_global=get_start_time()
is_plot = False
import gym
import numpy as np
import math  # In[2]:
import reinforcement_learning as rl  # In[3]:
tf.__version__  # In[4]:
gym.__version__

# ## Game Environment  # In[5]:
env_name = 'Breakout-v0'  # In[6]:
rl.checkpoint_base_dir = 'checkpoints_tutorial16/'  # In[7]:
rl.update_paths(env_name=env_name)

# ## Download Pre-Trained Model

# ## Create Agent  # In[9]:
agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=True,
                 use_logging=False)  # In[10]:
model = agent.model  # In[11]:
replay_memory = agent.replay_memory

# ## Training  # In[12]:
agent.run(num_episodes=1)

# ## Training Progress  # In[13]:
log_q_values = rl.LogQValues()
log_reward = rl.LogReward()  # In[14]:
log_q_values.read()
log_reward.read()

# ### Training Progress: Reward  # In[15]:
plt.plot(log_reward.count_states, log_reward.episode, label='Episode Reward')
plt.plot(log_reward.count_states, log_reward.mean, label='Mean of 30 episodes')
plt.xlabel('State-Count for Game Environment')
plt.legend()
if is_plot: plt.show()

# ### Training Progress: Q-Values  # In[16]:
plt.plot(log_q_values.count_states, log_q_values.mean, label='Q-Value Mean')
plt.xlabel('State-Count for Game Environment')
plt.legend()
if is_plot: plt.show()

# ## Testing  # In[17]:
agent.epsilon_greedy.epsilon_testing  # In[18]:
agent.training = False  # In[19]:
agent.reset_episode_rewards()  # In[20]:
agent.render = True  # In[21]:
agent.run(num_episodes=1)

# ### Mean Reward  # In[22]:
agent.reset_episode_rewards()  # In[23]:
agent.render = False  # In[24]:
agent.run(num_episodes=30)  # In[25]:
rewards = agent.episode_rewards
print("Rewards for {0} episodes:".format(len(rewards)))
print("- Min:   ", np.min(rewards))
print("- Mean:  ", np.mean(rewards))
print("- Max:   ", np.max(rewards))
print("- Stdev: ", np.std(rewards))  # In[26]:
_ = plt.hist(rewards, bins=30)

# ## Example States  # In[27]:
def print_q_values(idx):
    """Print Q-values and actions from the replay-memory at the given index."""
    q_values = replay_memory.q_values[idx]
    action = replay_memory.actions[idx]
    print("Action:     Q-Value:")
    print("====================")
    for i, q_value in enumerate(q_values):
        if i == action:
            action_taken = "(Action Taken)"
        else:
            action_taken = ""
        action_name = agent.get_action_name(i)
            
        print("{0:12}{1:.3f} {2}".format(action_name, q_value,
                                        action_taken))
    print()  # In[28]:
def plot_state(idx, print_q=True):
    """Plot the state in the replay-memory with the given index."""
    state = replay_memory.states[idx]
    
    fig, axes = plt.subplots(1, 2)
    ax = axes.flat[0]
    ax.imshow(state[:, :, 0], vmin=0, vmax=255,
              interpolation='lanczos', cmap='gray')
    ax = axes.flat[1]
    ax.imshow(state[:, :, 1], vmin=0, vmax=255,
              interpolation='lanczos', cmap='gray')
    if is_plot: plt.show()
    
    if print_q:
        print_q_values(idx=idx)  # In[29]:
num_used = replay_memory.num_used
num_used  # In[30]:
q_values = replay_memory.q_values[0:num_used, :]  # In[31]:
q_values_min = q_values.min(axis=1)
q_values_max = q_values.max(axis=1)
q_values_dif = q_values_max - q_values_min

# ### Example States: Highest Reward  # In[32]:
idx = np.argmax(replay_memory.rewards)
idx  # In[33]:
for i in range(-5, 3):
    plot_state(idx=idx+i)

# ### Example: Highest Q-Value  # In[34]:
idx = np.argmax(q_values_max)
idx  # In[35]:
for i in range(0, 5):
    plot_state(idx=idx+i)

# ### Example: Loss of Life  # In[36]:
idx = np.argmax(replay_memory.end_life)
idx  # In[37]:
for i in range(-10, 0):
    plot_state(idx=idx+i)

# ### Example: Greatest Difference in Q-Values  # In[38]:
idx = np.argmax(q_values_dif)
idx  # In[39]:
for i in range(0, 5):
    plot_state(idx=idx+i)

# ### Example: Smallest Difference in Q-Values  # In[40]:
idx = np.argmin(q_values_dif)
idx  # In[41]:
for i in range(0, 5):
    plot_state(idx=idx+i)

# ## Output of Convolutional Layers  # In[42]:
def plot_layer_output(model, layer_name, state_index, inverse_cmap=False):
    """
    Plot the output of a convolutional layer.
    :param model: An instance of the NeuralNetwork-class.
    :param layer_name: Name of the convolutional layer.
    :param state_index: Index into the replay-memory for a state that
                        will be input to the Neural Network.
    :param inverse_cmap: Boolean whether to inverse the color-map.
    """
    state = replay_memory.states[state_index]
    
    layer_tensor = model.get_layer_tensor(layer_name=layer_name)
    
    values = model.get_tensor_value(tensor=layer_tensor, state=state)
    num_images = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_images))
    fig, axes = plt.subplots(num_grids, num_grids, figsize=(10, 10))
    print("Dim. of each image:", values.shape)
    
    if inverse_cmap:
        cmap = 'gray_r'
    else:
        cmap = 'gray'
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    if is_plot: plt.show()

# ### Game State  # In[43]:
idx = np.argmax(q_values_max)
plot_state(idx=idx, print_q=False)

# ### Output of Convolutional Layer 1  # In[44]:
plot_layer_output(model=model, layer_name='layer_conv1', state_index=idx, inverse_cmap=False)

# ### Output of Convolutional Layer 2  # In[45]:
plot_layer_output(model=model, layer_name='layer_conv2', state_index=idx, inverse_cmap=False)

# ### Output of Convolutional Layer 3  # In[46]:
plot_layer_output(model=model, layer_name='layer_conv3', state_index=idx, inverse_cmap=False)

# ## Weights for Convolutional Layers  # In[47]:
def plot_conv_weights(model, layer_name, input_channel=0):
    """
    Plot the weights for a convolutional layer.
    
    :param model: An instance of the NeuralNetwork-class.
    :param layer_name: Name of the convolutional layer.
    :param input_channel: Plot the weights for this input-channel.
    """
    weights_variable = model.get_weights_variable(layer_name=layer_name)
    
    w = model.get_variable_value(variable=weights_variable)
    w_channel = w[:, :, input_channel, :]
    
    num_output_channels = w_channel.shape[2]
    w_min = np.min(w_channel)
    w_max = np.max(w_channel)
    abs_max = max(abs(w_min), abs(w_max))
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w_min, w_max))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w_channel.mean(),
                                                 w_channel.std()))
    num_grids = math.ceil(math.sqrt(num_output_channels))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_output_channels:
            img = w_channel[:, :, i]
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    if is_plot: plt.show()

# ### Weights for Convolutional Layer 1  # In[48]:
plot_conv_weights(model=model, layer_name='layer_conv1', input_channel=0)  # In[49]:
plot_conv_weights(model=model, layer_name='layer_conv1', input_channel=1)

# ### Weights for Convolutional Layer 2  # In[50]:
plot_conv_weights(model=model, layer_name='layer_conv2', input_channel=0)

# ### Weights for Convolutional Layer 3  # In[51]:
plot_conv_weights(model=model, layer_name='layer_conv3', input_channel=0)

# ## Discussion

# ## Exercises & Research Ideas

# ## License (MIT)
