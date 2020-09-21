import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = torch.Tensor(x) # Seems like this does nothing, even when numpy gets passed into Q
        layer_1 = self.l1(x)
        hidden = nn.functional.relu(layer_1) # Apply activation function as function rather than later
        output = self.l2(hidden)
        return output

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory)>self.capacity-1:
            del self.memory[0] # Would maybe be nice to store this for the case that memory.append fails
                               # but that requires quite extensive error handling which is not important here
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    # YOUR CODE HERE
    annealing_time = 1000
    progress = it/annealing_time
    
    max_eps = 1
    min_eps = 0.05
    epsilon = max(max_eps - (max_eps - min_eps) * progress, min_eps)
    
    return epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        # So we first need to choose whether we are taking a random action or a policy action
        if random.choices([True, False], weights=[self.epsilon, 1-self.epsilon], k=1)[0]:
            # This means we need to make a random choice for the action to be performed
            # The size of the output layer of Q_net is hardcoded as 2, so we will do that here too
            return random.choice(range(2))
        else:
            # This means we need to use the policy network
            obs = torch.Tensor(obs) # Stays Tensor if it was already one, becomes tensor if not
            action = torch.argmax(self.Q(obs)).item()
            return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


def compute_q_vals(Q, states, actions=None):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    all_actions = Q(states)
    if not actions is None:
        Q_values = all_actions[range(all_actions.shape[0]), actions.squeeze().tolist()].unsqueeze(dim=1)
    else: # If actions are not defined, we take the best action's Q-value
        Q_values, _ = all_actions.max(dim=1, keepdim=True)
        # Could be updated to include the argmax as well
    return Q_values
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    # YOUR CODE HERE
    
    # First, we need to find the max Q-value (over actions) from the next states
    future_Q_vals = compute_q_vals(Q, next_states)
    
    # if the next state is terminal, the Q-value should be zero:
    # turns out that 'dones' is not actually a boolean tensor, but an integer tensor. Waste of memory..
    # In codegrade, the dones are in fact boolean tensors. What a mess :')
    if dones.dtype == torch.bool: # Use boolean operators when actual boolean tensor
        done_tensor = ~dones
        future_Q_vals *= done_tensor
    else: # Must be a numerical tensor representing boolean values then
        done_tensor = 1 - dones  
        future_Q_vals *= done_tensor
    
    # With some complicated indexing tricks we could prevent the done states from passing through the Q-net
    # but this will likely not save a significant amount of processing time
    
    target = rewards + discount_factor * future_Q_vals
    
    return target

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            # So it seems like here we should sample an episode,
            # and every step update the weights
            
            # So first sample an action
            sampled_action = policy.sample_action(state)
            
            # Then step 
            state_tuple = env.step(sampled_action)
            
            # Store this transition in memory:
            s_next, r, done, _ = state_tuple
            memory.push((state, sampled_action, r, s_next, done))
            state = s_next
            
            # Now that we have added a transition, we should try to train based on our memory
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            # This is like online learning, we could also only train once per episode
            
            steps += 1
            global_steps += 1
            
            # Update epsilon
            policy.set_epsilon(get_epsilon(global_steps))
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                    print("epsilon: ", policy.epsilon)
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations
