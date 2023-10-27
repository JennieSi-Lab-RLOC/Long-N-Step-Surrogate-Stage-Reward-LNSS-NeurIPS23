import numpy as np
import torch
import gym
import argparse
import os


from collections import deque
import utils
import replay_buffer
import TD3
import D4PG
import OurDDPG

import csv

from mpi4py import MPI
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=5,n_worker=8):
	eval_env = gym.make(env_name)
	print(seed)
	eval_env.seed(seed + 100)
	eval_reward = []
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		eval_r = 0
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
			eval_r += reward
			if done:
				eval_reward.append(eval_r)

	avg_reward /= eval_episodes
	global_total_reward = MPI.COMM_WORLD.allreduce(avg_reward, op=MPI.SUM)
	avg_reward = global_total_reward/n_worker
	std_eval = np.std(eval_reward)
	
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward,std_eval


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="Hopper-v2")          	# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=8e3, type=int) # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=8e5, type=int) # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)		# Batch size for both actor and critic
	parser.add_argument("--buffer_size", default=1e6, type=int)     # buffer size 
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--N_step", default=1, type=int)            # N step return
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--n_worker", default=1,type=int)           # number of worker
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--name", default="")                       # add name to csv file
	parser.add_argument("--n_run", default=1, type=int)			    # number of sequential run
	args = parser.parse_args()

for num_run in range(int(args.n_run)):
	file_name = f"{args.policy}_{args.env}_{args.seed}_{num_run}_{args.N_step}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed + num_run}, N step: {args.N_step}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	seed_num = args.seed + num_run
	env.seed(seed_num + MPI.COMM_WORLD.Get_rank())
	env.action_space.seed(seed_num + MPI.COMM_WORLD.Get_rank())
	np.random.seed(seed_num + MPI.COMM_WORLD.Get_rank())
	torch.manual_seed(seed_num + MPI.COMM_WORLD.Get_rank())
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

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
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = D4PG.D4PG(**kwargs)
	

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = replay_buffer.ReplayBuffer(state_dim, action_dim, int(args.buffer_size))
	

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	
	training_data = []
	huris_ep_reward =[]
	exp_buffer = deque()
	N_step_number = args.N_step
	
	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
		
		#make reward be positive definite
		if args.env == "Hopper-v2":
			reward_mod = reward + 1.5
		elif args.env == "Walker2d-v2":
			reward_mod = reward + 2
		else:
			reward_mod = reward
		
		if reward_mod < 0:
			reward_mod = 0

		exp_buffer.append((state, action, reward_mod,next_state,done_bool))
		
		
		# Store data in replay buffer
		if len(exp_buffer) >= N_step_number:
			state_0, action_0, reward_0,next_state_1,done_1 = exp_buffer.popleft()
			discounted_reward = reward_0
			gamma = args.discount
			for (_, _, r_i, _, _) in exp_buffer:
				discounted_reward += r_i * gamma
				gamma *= args.discount
			#store data in memory buffer
			ds_factor = (args.discount - 1)/(gamma - 1)
			discounted_reward = ds_factor * discounted_reward
			replay_buffer.add((state_0, action_0, discounted_reward,next_state_1,(1 - done_1),args.discount))


		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)
			
		if done:
			
			#store rest of experiences remaining in buffer
			while len(exp_buffer) != 0:
				state_0, action_0, reward_0,next_state_1,done_1 = exp_buffer.popleft()
				discounted_reward = reward_0
				gamma = args.discount
				for (_, _, r_i, _, _) in exp_buffer:
					discounted_reward += r_i * gamma
					gamma *= args.discount
					
				ds_factor = (args.discount - 1)/(gamma - 1)
				discounted_reward = ds_factor * discounted_reward	
				replay_buffer.add((state_0, action_0,discounted_reward, next_state_1,(1 - done_1),args.discount))
			
			
			exp_buffer.clear()
			huris_ep_reward.append(episode_reward)
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			eval_reward,std = eval_policy(policy, args.env, seed_num,eval_episodes=5,n_worker = args.n_worker)

			if args.save_model: policy.save(f"./models/{file_name}")
			avg_training_reward = np.array(huris_ep_reward[-1:-20:-1]).mean()
			std_training = np.std(np.array(huris_ep_reward[-1:-20:-1]))
			
			
			training_data.append([(t + 1),avg_training_reward,std_training,eval_reward,std])


	fields = ['Epois number','mean training Epois reward','Std Epois reward','eval Epois reward','std eval']
	filename = f"{args.policy}_{args.env}_{seed_num}_{args.N_step}_{args.name}.csv"
	with open(filename, 'w') as csvfile: 
	    # creating a csv writer object 
	    csvwriter = csv.writer(csvfile,lineterminator = '\n') 
	    # writing the fields 
	    csvwriter.writerow(fields) 
	    # writing the data rows 
	    csvwriter.writerows(training_data)
