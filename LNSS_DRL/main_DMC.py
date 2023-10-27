import numpy as np
import torch
import argparse
import os
from dm_control import suite
from dm_control import viewer
from collections import deque

import replay_buffer as buffer
import TD3
import OurDDPG
import D4PG

import csv
from mpi4py import MPI
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, domain_name, task_name, seed, eval_episodes=5,n_worker=4):
	environment_kwargs = {'flat_observation': True}
	eval_env = suite.load(domain_name=args.domain, 
			task_name=args.task, 
			environment_kwargs=environment_kwargs, 
			task_kwargs={'random': (seed + 100)})
	print(seed)
	avg_reward = 0.
	eval_reward = []
	epois_reward = 0.
	for _ in range(eval_episodes):
		epois_reward = 0.
		state_type, reward, discount, state = eval_env.reset()
		done = False
		while not done:
			action = policy.select_action(state['observations'])
			step_type, reward, discount, state = eval_env.step(action)
			
			done = step_type.last()
			avg_reward += reward
			epois_reward += reward
			if done:
				eval_reward.append(epois_reward)

	avg_reward /= eval_episodes
	global_avg_reward = MPI.COMM_WORLD.allreduce(avg_reward, op=MPI.SUM)
	#devide number of agent use
	
	global_avg_reward = global_avg_reward/n_worker
	std_eval = np.std(eval_reward)
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return global_avg_reward,std_eval


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="D4PG")                  # Policy name (TD3)
	parser.add_argument("--domain", default="cartpole")	   			# DeepMind Control Suite domain name
	parser.add_argument("--task", default="swingup")                # DeepMind Control Suite task name
	parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate 1e4
	parser.add_argument("--max_timesteps", default=1e5, type=int)   # Max time steps to run environment  9e5
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--buffer_size", default=20, type=int) 	    # Memory buffer size
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--N_step", default=1, type=int)            # N step return
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
	parser.add_argument("--reward_noise", default=0, type=float)    # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)    # Range to clip target policy noise
	parser.add_argument("--n_worker", default=1,type=int)			# number of worker
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--n_run", default=5, type=int)			    # number of sequential run
	args = parser.parse_args()
	
	
	BUFFER_SIZE = 2**args.buffer_size
	for num_run in range(args.n_run):
	
		file_name = f"{args.policy}_{args.domain}_{args.task}_{args.seed}"
		print("---------------------------------------")
		print(f"Policy: {args.policy}, Domain: {args.domain}, Task: {args.task}, Seed: {args.seed + num_run}, N step: {args.N_step}")
		print("---------------------------------------")
	
		if not os.path.exists("./results"):
			os.makedirs("./results")
	
		if args.save_model and not os.path.exists("./models"):
			os.makedirs("./models")
			
		# set random seeds for reproduce	
		seed_num = args.seed + num_run
		environment_kwargs = {'flat_observation': True}
		env = suite.load(domain_name=args.domain, 
						task_name=args.task, 
						environment_kwargs=environment_kwargs, 
						task_kwargs={'random': (seed_num + MPI.COMM_WORLD.Get_rank())})
		np.random.seed(seed_num + MPI.COMM_WORLD.Get_rank())
		torch.manual_seed(seed_num + MPI.COMM_WORLD.Get_rank())
		
		state_dim = env.observation_spec()['observations'].shape[0]
		action_dim = env.action_spec().shape[0] 
		max_action = float(env.action_spec().maximum[0])
		min_action = float(env.action_spec().minimum[0])
		action_shape = env.action_spec().shape
	
		kwargs = {
			"state_dim": state_dim,
			"action_dim": action_dim,
			"max_action": max_action,
			"discount": args.discount,
			"tau": args.tau,
		}
	
		# Initialize policy
		if args.policy == "TD3":
			# Target policy smoothing is scaled wrt the action scale
			kwargs["policy_noise"] = args.policy_noise * max_action
			kwargs["noise_clip"] = args.noise_clip * max_action
			kwargs["policy_freq"] = args.policy_freq
			policy = TD3.TD3(**kwargs)
		elif args.policy == "DDPG":
			policy = OurDDPG.DDPG(**kwargs)
		elif args.policy == "D4PG":
			# Target policy smoothing is scaled wrt the action scale
			kwargs["policy_noise"] = args.policy_noise * max_action
			kwargs["noise_clip"] = args.noise_clip * max_action
			kwargs["policy_freq"] = args.policy_freq
			policy = D4PG.D4PG(**kwargs)
	
		if args.load_model != "":
			policy_file = file_name if args.load_model == "default" else args.load_model
			policy.load(f"./models/{policy_file}")
		
		
		replay_buffer = buffer.ReplayBuffer(state_dim, action_dim, BUFFER_SIZE)
		

		state_type, reward, discount, state = env.reset()
		done = False
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0
	
		training_data = []
		huris_ep_reward = []
		
		exp_buffer = deque()
		N_step_number = args.N_step
		
		state_type, reward, discount, state = env.reset()
		done = False
		exp_buffer.clear()
		episode_timesteps = 0	
		actual_Q = 0
		episode_reward = 0
		
		for t in range(int(args.max_timesteps)):
			
			episode_timesteps += 1
	
			# Select action randomly or according to policy
			if t < args.start_timesteps:
				action = np.random.uniform(low=min_action, high=max_action, size=action_shape)
			else:
				action = (
					policy.select_action(state['observations'])
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)
	
			# Perform action
			step_type, reward, discount, next_state = env.step(action) 
			
			done = step_type.last()
			
			#ADD NOISE TO REWARD:
			noise_r = np.clip(np.random.uniform(-1, 1) * args.reward_noise , -args.noise_clip, args.noise_clip)
			wnr = np.clip(reward + noise_r , 0, 1)
			exp_buffer.append((state['observations'], action,
							 wnr,next_state['observations'],done))
	
			# Compute LNSS reward and Store data in replay buffer
			if len(exp_buffer) >= N_step_number:
				state_0, action_0, reward_0,next_state_1,done_1 = exp_buffer.popleft()
				discounted_reward = reward_0
				gamma = args.discount
				for (_, _, r_i, _, _) in exp_buffer:
					discounted_reward += r_i * gamma
					gamma *= args.discount
				#apply LNSS discounted factor to reward
				ds_factor = (args.discount - 1)/(gamma - 1)
				discounted_reward = ds_factor * discounted_reward
				#store data in memory buffer
				replay_buffer.add((state_0, action_0, discounted_reward,next_state_1,(1 - done_1),args.discount))
	
			state = next_state
			episode_reward += reward
			# Train agent after collecting sufficient data
			if t >= args.start_timesteps:
				policy.train(replay_buffer, args.batch_size,t)
				
	
			if done:
				#store rest of experiences remaining in buffer
				while len(exp_buffer) != 0:
					state_0, action_0, reward_0,next_state_1,done_1 = exp_buffer.popleft()
					discounted_reward = reward_0
					gamma = args.discount
					for (_, _, r_i, _, _) in exp_buffer:
						discounted_reward += r_i * gamma
						gamma *= args.discount
					#apply LNSS discounted factor to reward
					ds_factor = (args.discount - 1)/(gamma - 1)
					discounted_reward = ds_factor * discounted_reward	
					
					replay_buffer.add((state_0, action_0,discounted_reward, next_state_1,(1 - done_1),args.discount))
				
				exp_buffer.clear()
				huris_ep_reward.append(episode_reward)

				
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
				state_type, reward, discount, state = env.reset()
				done = False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1

	
			# Evaluate episode
			if (t + 1) % args.eval_freq == 0:
				eval_reward,eval_std = eval_policy(policy, args.domain, args.task, seed_num,eval_episodes=5,n_worker = args.n_worker)
				avg_training_reward = np.array(huris_ep_reward).mean()
				std_training = np.std(np.array(huris_ep_reward))
				training_data.append(((t + 1),avg_training_reward,std_training,eval_reward,eval_std))
				if args.save_model: policy.save(f"./models/{file_name}")
	
		fields = ['Epois number','mean training Epois reward','Std Epois reward',
				'eval Epois reward','eval std']
		filename = f"{args.policy}_{args.domain}_{args.task}_{seed_num}.csv"
		with open(filename, 'w') as csvfile: 
		    # creating a csv writer object 
		    csvwriter = csv.writer(csvfile,lineterminator = '\n') 
		
		    # writing the fields 
		    csvwriter.writerow(fields) 
		
		    # writing the data rows 
		    csvwriter.writerows(training_data)
