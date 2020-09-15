import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

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
        
        # Get number of possible actions
        num_actions = len(self.Q[obs])

        # Get index of action corresponding to maximum Q-value
        max_indices = [index for index, val in enumerate(self.Q[obs]) if val == max(self.Q[obs])]
        
        # break ties consistently
        max_index = max_indices[0]
        
        # Probabilities
        sample_probs = num_actions * [self.epsilon / num_actions]
        sample_probs[max_index] += (1 - self.epsilon)
        
        # sample
        action = np.random.choice(4, 1, p = sample_probs)
        
        return int(action)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        # Following the Figure in Example 6.5, we initialize the starting state at s = 30
        state = env.reset()
        
        # boolean for determining whether episode is finished (terminal state reached)
        done = False
        
        # choose A from S using epsilon-greedy policy, derived from Q
        action = policy.sample_action(state)
             
        # loop until terminal state reached
        while not done:
            
            # take current action, observe current reward and next state
            next_state, r, done, _ = env.step(action)
            
            # choose next action from next state using policy derived from Q
            next_action = policy.sample_action(next_state)
            
            # update Q
            Q[state, action] = Q[state, action] + alpha * (r + discount_factor * Q[next_state, next_action] - Q[state, action])
            
            # update current steps and actions
            state = next_state
            action = next_action
            
            # update statistics
            i += 1
            R += r
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        # initialize state
        state = env.reset()
        
        # boolean for determining when episode has ended
        done = False
        
        # loop until s is terminal state
        while not done:
            
            # choose action from state using policy derived from Q (epsilon-greedy in this case)
            action = policy.sample_action(state)
            
            # take action, and observe corresponding reward and next state
            next_state, reward, done, _ = env.step(action)
            
            max_Q = [Q_val for Q_val in Q[next_state] if Q_val == max(Q[next_state])][0]
            
            # update rule
            Q[state, action] = Q[state, action] + alpha * (reward + (discount_factor * max_Q) - Q[state, action])
            
            state = next_state
            
            # update statistics
            i += 1
            R += reward
            
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)
