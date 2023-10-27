# A Long N-step Surrogate Stage Reward for Deep Reinforcement Learning (LNSS)
Official implementation of LNSS in Deepmind control suite and OpenAI GYM.


**[A Long N-step Surrogate Stage Reward for Deep Reinforcement Learning](https://nips.cc/virtual/2023/poster/72325)**

NeurIPS | 2023

[Junmin Zhong](https://scholar.google.com/citations?user=uVv_eWQAAAAJ&hl=en&oi=ao)<sup>1</sup>, Ruofan Wu<sup>1</sup>, Jennie Si<sup>1</sup>

<sup>1</sup>Arizona State University

## Installation

The code has been tested on Python 3.10.6 and PyTorch 2.0.1. 

See other required packages:
  1. [Mujoco-py](https://github.com/openai/mujoco-py)
  2. [Deepmind Control Suite](https://github.com/google-deepmind/dm_control)
  3. [OpenAI GYM](https://github.com/openai/gym)
  4. [mpi4py](https://pypi.org/project/mpi4py/)
  5. Numpy

## Q learning Example

We first show how LNSS can benefit learning by using a simple Maze environment. To provide insight on the effect of using LNSS, we compare the performance of the original $Q$-learning with the one that the stage reward is replaced by LNSS reward. 
[Maze Environment and results](Q_result.png)

To run the experiment: 
```python3 run_main.py```


Note: penalty = 0 for no penalty and penalty = 1 for penalty. N = 1 for the original reward. N > 1 for LNSS and N = 5 for best result.

## DRL Code base

TD3 and DDPG are based on [Auther's Code](https://github.com/sfujim/TD3). D4PG is our modified code based on TD3 and other [pytorch implementation](https://github.com/schatty/d4pg-pytorch).

## Training

Example Run:

Target policy is TD3, LNSS N parameter is 100 and use 8 parallel actors to train:

```mpirun -n 8 python3 main_DMC.py --policy="TD3" --N_step=100 --n_worker=8```
note: if current os is windows, please use mpiexec

## Discrete Cheetah Run

We also provide our annotated cheetah file for reference. If you want to run, please replace the original DMC cheetah.py with ours.
