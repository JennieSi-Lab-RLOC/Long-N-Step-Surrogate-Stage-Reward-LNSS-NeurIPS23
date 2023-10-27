import numpy as np
import pandas as pd


class Qlearning:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1,Seed = 0):
        np.random.seed(Seed)
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Q-Learning"
        print("Using Q-Learning ...")

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation,epoch):
        # Add non-existing state to our q_table
        self.check_state_exist(observation)
        epsilon = self.epsilon**(epoch/100)
        # Select next action
        if np.random.uniform() >= epsilon:
            # Choose argmax action
            state_action_values = self.q_table.loc[observation, :]
            action = np.random.choice(state_action_values[state_action_values == np.max(state_action_values)].index) # handle multiple argmax with random
        else:
            # Choose random action
            action = np.random.choice(self.actions)

        return action


    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max() # max state-action value
        else:
            q_target = r  # next state is terminal

        self.q_table.loc[s, a] = self.q_table.loc[s, a] + self.lr * (q_target - self.q_table.loc[s, a])
        return self.q_table
        


    '''States are dynamically added to the Q(S,A) table as they are encountered'''
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
