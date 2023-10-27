from maze_env import Maze
from RL_brain_q_learning import Qlearning as QL
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import time


from collections import deque
import csv

DEBUG=1
def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg) 
        else:
            print(msg) 



def update(env, RL, data, episodes=50,N_step_number=5):
    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward
    data['final_Q']=[]
    gamma = 0.9

    for episode in range(episodes):  
        t=0
        exp_buffer = deque()
        # initial state
        if episode == 0:
            state = env.reset(value = 0)
        else:
            state = env.reset()
       
        debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
        while True:
            if(showRender or (episode % renderEveryNth)==0):
                env.render(sim_speed)
            action = RL.choose_action(str(state),episode)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            exp_buffer.append((state, action, reward,state_,done))
            global_reward[episode] += reward
            debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2,'reward_{}=  total return_t ={} Mean50={}'.format(reward, global_reward[episode],np.mean(global_reward[-50:])))
            

            # LNSS REWARD
            if len(exp_buffer) >= N_step_number:
                state1, action1, reward1, state_1,_ = exp_buffer.popleft()
                discount_factor = gamma
                discounted_reward = reward1
                for (_, _, r_i, _, _) in exp_buffer:
                    discounted_reward += r_i * discount_factor
                    discount_factor *= gamma
                ds_factor = (gamma - 1)/(discount_factor - 1)
                discounted_reward = ds_factor * discounted_reward
                Q =  RL.learn(str(state1), action1, discounted_reward, str(state_1))

            state = state_
            # break while loop when end of this episode
            if done or t > 50:
                while len(exp_buffer) != 0:
                    state1, action1, reward1, state_1,_ = exp_buffer.popleft()
                    discount_factor = gamma
                    discounted_reward = reward1
                    for (_, _, r_i, _, _) in exp_buffer:
                        discounted_reward += r_i * discount_factor
                        discount_factor *= gamma
                    
                    ds_factor = (gamma - 1)/(discount_factor - 1)
                    discounted_reward = ds_factor * discounted_reward
                    
                    Q =  RL.learn(str(state1), action1, discounted_reward, str(state_1))
                    
                break
            else:
                t=t+1

        debug(1,"({}) Episode {}: Length={}  Total return = {} ".format(RL.display_name,episode, t,  global_reward[episode],global_reward[episode]),printNow=(episode%printEveryNth==0))
        if(episode>=100):
            debug(1,"    Median100={} Variance100={}".format(np.median(global_reward[episode-100:episode]),np.var(global_reward[episode-100:episode])),printNow=(episode%printEveryNth==0))
    # end of game
    print('game over -- Algorithm {} completed'.format(RL.display_name))
    data['final_Q']= Q
    env.destroy()

if __name__ == "__main__":
    sim_speed = 0.05
    
    #########2 experiment parameters
    N_step_number = 5 ###N for LNSS, N = 1 for original reward
    penalty = 1 ### 1 for penalty, 0 for no penalty
    
    #Exmaple Full Run, you may need to run longer
    showRender=False
    episodes=600
    renderEveryNth=10000
    printEveryNth=100
    do_plot_rewards=True

    if(len(sys.argv)>1):
        episodes = int(sys.argv[1])
    if(len(sys.argv)>2):
        showRender = sys.argv[2] in ['true','True','T','t']
    if(len(sys.argv)>3):
        datafile = sys.argv[3]


    #All Tasks
    agentXY=[0,0]
    goalXY=[1,2]
    wall_shape=np.array([[1,1],[0,2],])
    pits=np.array([])
    experiments = []
    
    n_run = 10
    for n in range(n_run):
        # alg1 (Q-Learning)
        seed = n
        np.random.seed(seed)
        env1 = Maze(agentXY,goalXY,wall_shape, pits, penalty = penalty)
        qlearning = QL(actions=list(range(env1.n_actions)),Seed= seed)
        data1={}
        env1.after(10, update(env1, qlearning, data1, episodes,N_step_number))
        env1.mainloop()
        experiments.append((env1,RL1, data1))
        
        fields = ['Epois reward']
        filename = f"{n}_lnss_{n_rollout}.csv"
        with open(filename, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile,lineterminator = '\n') 
        
            # writing the fields 
            csvwriter.writerow(fields) 
        
            # writing the data rows 
            csvwriter.writerows(data1['global_reward'].reshape((episodes, 1)))
            
        file_path = f"{n}_Q_{n_rollout}.csv"   
        data1['final_Q'].to_csv(file_path)
    
        print("All experiments complete")
    
        for env, RL, data in experiments:
            print("{} : max reward = {} medLast100={} varLast100={}".format(RL.display_name, np.max(data['global_reward']),np.median(data['global_reward'][-100:]), np.var(data['global_reward'][-100:])))



