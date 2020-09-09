import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    
    # Outer loop loops while accuracy of estimation (i.e., function change) exceeds threshold theta for all states
    #for k in range(1000):
    while True:
        
        # set function change to zero so it gets updated for a small change
        function_change = 0.0
        
        # Inner loop loops over each state (excluding the terminal states)
        for s in range(env.nS):
            
            # save current state-values
            v = V[s]
            
            # get lists of transition probabilities, successive states, and corresponding rewards for all actions
            probs = [env.P[s][a][0][0] for a in range(env.nA)]
            next_states = [env.P[s][a][0][1] for a in range(env.nA)]
            rewards = [env.P[s][a][0][2] for a in range(env.nA)]
            
            # Policiy evaluation update step ( based on equation 4.5)
            V[s] = np.sum(policy[s] * (probs * (rewards + discount_factor * V[next_states])))
            
            # Update function change
            function_change = np.maximum(function_change, abs(v - V[s]))
            
        # Converged if function change is smaller than theta for all states
        if function_change < theta:
            break
    
    return np.array(V)
# Added by Lennert:
import pdb
from copy import deepcopy

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    
    # Start with V all zeros
    V = np.zeros((env.nS))
    
    while True:
        # Should consist of two steps:
        # 1. Policy evaluation
        V = policy_eval_v(policy, env, discount_factor)
        
        # 2 - policy improvement step
        # iterate over each step
        policy_old = deepcopy(policy)
        policy_stable = True
        for s in range(env.nS):
            
            old_action = deepcopy(policy[s])
            
            # get lists of transition probabilities, successive states, and corresponding rewards for all actions
            probs = [env.P[s][a][0][0] for a in range(env.nA)]
            next_states = [env.P[s][a][0][1] for a in range(env.nA)]
            rewards = [env.P[s][a][0][2] for a in range(env.nA)]
            
            # compute expected returns for current state over all possible actions
            expected_returns = probs * (rewards + discount_factor * V[next_states])
            
            # select action_index corresponding to highest expected return
            max_indices = [idx for idx, val in enumerate(expected_returns) if val == max(expected_returns)]
            
            # break ties randomly
            max_index = np.random.choice(max_indices) 
            
            # update policy
            policy[s] = [1 if index == max_index else 0 for index in range(env.nA)]
            
            # policy is unstable if it changed after update
            if np.argmax(old_action) != np.argmax(policy[s]):
                policy_stable = False
                
        if policy_stable:
            break
                   
    return policy, V
from copy import deepcopy
import pdb
def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    
    delta = -1
    policy = np.ones([env.nS, env.nA]) / env.nA
    while (delta>theta or delta < 0):
        old_Q = deepcopy(Q)
        for s in range(env.nS):
            probs = np.array([env.P[s][a][0][0] for a in range(env.nA)])
            next_states = np.array([env.P[s][a][0][1] for a in range(env.nA)])
            rewards = np.array([env.P[s][a][0][2] for a in range(env.nA)])

            # Maximum action-value over actions in the next state
            # This reflects the greedy, deterministic policy that we implement during value iteration:
            # take the action at each step that has the highest action-value
            next_Q = np.array([max(Q[next_state]) for next_state in next_states])
            
            Q[s] = probs * (rewards + discount_factor * next_Q)

            # Based on the following equation:
            # Q[s][a] = "max over a'""sum over s' and r" probs * (rewards + discount_factor * Q[next_states])
            # Sum over s' (next states) and r drops out, as for each action there is a single known 
            # next state and reward. (still weird that the 0 reward only is given when an action is
            # taken from a terminal state)
            # max over a': maximum over actions taken from the next state, included in next_Q
            # probs: probability of each transition to next state
            # rewards: reward of action taken now
            # next_Q: list of (env.nA) that for each entry gives the action-value of the ideal action
            
        delta = np.max(abs(Q-old_Q)) #Find largest difference between new and old Q

    # For this exercise, we should not generate a stochastic policy, but a one-hot deterministic policy.
    # Stochastic policy could be done by taking the softmax of the Q array along dim 1.
    
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        action_choice = np.argmax(Q[s,:])
        policy[s,action_choice] = 1
    
    return policy, Q
