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
    # NB: terminal states (V[0] and V[env.nS - 1] must always be valued at zero)
    V = np.zeros(env.nS)
    
    # Outer loop loops while accuracy of estimation (i.e., function change) exceeds threshold theta for all states
    for k in range(1000):
        
        # set function change to zero so it gets updated for a small change
        function_change = 0.0
        
        # Inner loop loops over each state (excluding the terminal states)
        for s in range(1, env.nS - 1):
            
            # save current state-values
            v = V[s]
            
            # get lists of transition probabilities, successive states, and corresponding rewards for all actions
            probs = [env.P[1][a][0][0] for a in range(env.nA)]
            next_states = [env.P[1][a][0][1] for a in range(env.nA)]
            rewards = [env.P[1][a][0][2] for a in range(env.nA)]
            
            # Policiy evaluation update step ( based on equation 4.5)
            V[s] = np.sum(policy[s] * (probs * (rewards + discount_factor * V[next_states])))
            
            # Update function change
            function_change = np.maximum(function_change, abs(v - V[s]))
            
        # Converged if function change is smaller than theta for all states
        if function_change < theta:
            break
    
    return np.array(V)
