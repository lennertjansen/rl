import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        
        # So we need to determine for every input state-action pair, what the resulting policy distribution is
        # This means that the input will be a single state and a single action per index. 
        # We then need to determine if, according to our policy, the action should be taken (prob=1) 
        # or not (prob=0)
        
        # state is a tuple of (player's current sum, dealer's single showing card, boolean for usable ace)
        probs = []
        for index, (state, action) in enumerate(zip(states, actions)):
            chosen_action = self.sample_action(state)
            if action == chosen_action:
                probs.append(1)
            else:
                probs.append(0)
                    
        
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        if state[0] == 20 or state[0] == 21: #Now we should stick (0)
            action = 0
        else: # Otherwise hit
            action = 1
    
        return action

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        Hint: Do not include the state after the termination in the list of states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    done = False
    state = env.reset() # Could also use env._get_obs(), but codegrade seems to expect this
    while done == False:
        states.append(state)
    
        action = policy.sample_action(state)
        actions.append(action)
    
        state, reward, done, _ = env.step(action)
        
        rewards.append(reward)
        dones.append(done)

    return states, actions, rewards, dones

from collections import Counter
import gym

def mc_prediction(env, policy, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    # YOUR CODE HERE    
        
    # Due to the structure of the gym environment, it is not trivial to map the entire state space
    # so we only map the state space of the BlackJack env
    count_zeros = False
    if (isinstance(env.observation_space, gym.spaces.tuple_space.Tuple)):
        if (len(env.observation_space.spaces) == 3):
            count_zeros = True
        
        
    state_tuples = [(first, second, bool(third)) for first in range(2,env.observation_space.spaces[0].n)
                       for second in range(1,env.observation_space.spaces[1].n)
                       for third in range(env.observation_space.spaces[2].n)]
    returns = {state_tuple: [] for state_tuple in state_tuples}
    
    if count_zeros:
        # Replace the returns_count with a Counter object, and fill with all possible states
        returns_count = Counter({state_tuple: 0 for state_tuple in state_tuples})
    
    
    for episode in tqdm(range(num_episodes)): # num_episodes
        
        env.reset()
        states, actions, rewards, dones = sampling_function(env, policy)
        p_return = 0
        
        for index in reversed(range(len(states))): # Reverse so we loop in opposite direction through timesteps
            c_state = states[index]
            c_action = actions[index]
            c_reward = rewards[index]

            p_return = discount_factor * p_return + c_reward
                        
            if len(returns[c_state]) == 0:
                returns[c_state] = [p_return]
            else:
                returns[c_state].append(p_return)
            
            if count_zeros:
                returns_count[c_state] += 1
    
    V = {state: np.nan_to_num(np.mean(value)) for (state, value) in returns.items()}
    if count_zeros:
        zero_counts = [True for item in list(returns_count) if returns_count[item] == 0]
    
        no_of_zero = sum(zero_counts)
        if no_of_zero>0:
            print(f"Did not reach {no_of_zero} states in MC estimation. Value estimation for these states is missing.")
        else:
            print("Reached all states in MC estimation.")
        
    return V

class RandomBlackjackPolicy(object):
    """
    A random BlackJack policy.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        
        probs = np.ones(len(states))/2
        return probs
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        
        action = np.random.choice(1)
        
        
        return action
import gym

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    # YOUR CODE HERE
    
    epsilon = 1e-6
    
        
    # Due to the structure of the gym environment, it is not trivial to map the entire state space
    # so we only map the state space of the BlackJack env
    count_zeros = False
    if (isinstance(env.observation_space, gym.spaces.tuple_space.Tuple)):
        if (len(env.observation_space.spaces) == 3):
            count_zeros = True
        
    state_tuples = [(first, second, bool(third)) for first in range(2,env.observation_space.spaces[0].n)
                       for second in range(1,env.observation_space.spaces[1].n)
                       for third in range(env.observation_space.spaces[2].n)]
    returns = {state_tuple: [] for state_tuple in state_tuples}
    
    if count_zeros:
        returns_count = Counter({state_tuple: 0 for state_tuple in state_tuples})
    
    for episode in tqdm(range(num_episodes)): # num_episodes
        
        env.reset()
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        p_return = 0
        
        pi = target_policy.get_probs(states, actions)
        b = (behavior_policy.get_probs(states, actions) + epsilon)
        pi_div_b = target_policy.get_probs(states, actions) / (behavior_policy.get_probs(states, actions) + epsilon)

        for index in reversed(range(len(states))): # Reverse so we loop in opposite direction through timesteps
            c_state = states[index]
            c_action = actions[index]
            c_reward = rewards[index]

            p_return = discount_factor * p_return + c_reward
            W = np.cumprod(pi_div_b[index:])
            
            p_return = W[0] * p_return
            if len(returns[c_state]) == 0:
                returns[c_state] = [p_return]
            else:
                returns[c_state].append(p_return)

            if count_zeros:
                returns_count[c_state] += 1
    
    V = {state: np.nan_to_num(np.mean(value)) for (state, value) in returns.items()}
    
    if count_zeros:
        zero_counts = [True for item in list(returns_count) if returns_count[item] == 0]
        no_of_zero = sum(zero_counts)
        if no_of_zero>0:
            print(f"Did not reach {no_of_zero} states in MC estimation. Value estimation for these states is missing.")
        else:
            print("Reached all states in MC estimation.")
    
    return V
