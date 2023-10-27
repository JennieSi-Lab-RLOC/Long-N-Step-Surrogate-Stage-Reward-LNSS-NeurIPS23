import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from mpi_utils.normalizer import normalizer
from torch.distributions import Categorical
from l2_projection import _l2_project

device = torch.device("cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim,v_min,v_max,num_atoms):
		"""
		Args:
			num_states (int): state dimension
			num_actions (int): action dimension
			hidden_size (int): size of the hidden layers
			v_min (float): minimum value for critic
			v_max (float): maximum value for critic
			num_atoms (int): number of atoms in distribution
			init_w:
		"""
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, num_atoms)
		
		self.z_atoms = np.linspace(v_min, v_max, num_atoms)



	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		return q1
	
	def get_probs(self, state, action):
		q1 = self.forward(state, action)
		q1 = torch.softmax(q1, dim=1)
		return q1
	
	


class D4PG(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		v_min = 0,
		v_max = 100,
		num_atoms = 51,
		
	):
		self.adam_lr = 3e-4
		#distributional learning
		self.v_min = v_min
		self.v_max = v_max
		self.num_atoms = num_atoms
		self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		sync_networks(self.actor)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.adam_lr)

		self.critic = Critic(state_dim, action_dim, self.v_min, self.v_max, self.num_atoms).to(device)
		sync_networks(self.critic)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.adam_lr)
		
		#the Binary Cross Entropy between the target and the input probabilities
		self.value_criterion = nn.BCELoss(reduction='none')

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
		
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()
	


	def train(self, replay_buffer, batch_size=256,T = 1):
		self.total_it += 1
		####Get experience
		state, action, next_state, reward, not_done,d_gamma = replay_buffer.sample(batch_size)

		
		#start update
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# predict Z distribution with target Z network
			target_Q = self.critic_target.get_probs(next_state, next_action)
		
			#projected distribution
			target_Z_projected = _l2_project(next_distr_v=target_Q,
                                         rewards_v=reward,
                                         dones_mask_t=(1-not_done),
                                         gamma=self.discount,
                                         n_atoms=self.num_atoms,
                                         v_min=self.v_min,
                                         v_max=self.v_max,
                                         delta_z=self.delta_z)
			#trans to tensor
			target_Z_projected = torch.from_numpy(target_Z_projected).float().to(device)
			
			
		# Get current Q estimates
		current_Q = self.critic.get_probs(state, action)
		
		# Compute critic loss
		critic_loss = self.value_criterion(current_Q, target_Z_projected).mean(axis=1) 
		
		# Optimize the critic
		critic_loss = critic_loss.mean()
		
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		sync_grads(self.critic)
		self.critic_optimizer.step()
		
		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			actor_loss = self.critic.get_probs(state, self.actor(state))
			actor_loss = actor_loss*torch.from_numpy(self.critic.z_atoms).float().to(device)
			actor_loss = torch.sum(actor_loss,dim=1)
			actor_loss = -actor_loss.mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			sync_grads(self.actor)
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			
	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		
