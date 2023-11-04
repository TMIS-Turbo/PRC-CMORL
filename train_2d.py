import argparse
import torch
import numpy as np
import pandas as pd
import random as rn
import gym
import math
import matplotlib.pyplot as plt
import environments
import cmo_ddpg
from ounoise import OrnsteinUhlenbeckActionNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from index_2d import compute_index
from utils import save_model
import os

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="MO-Swimmer-v2", help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.01, metavar='G', help='discount factor for model (default: 0.01)')
parser.add_argument('--no_ou_noise', default=False, action='store_true')
parser.add_argument('--param_noise', default=True, action='store_true')
parser.add_argument('--noise_scale', type=float, default=0.2, metavar='G', help='(default: 0.2)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size (default: 64)')
parser.add_argument('--param_noise_interval', type=int, default=50, metavar='N')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N', help='number of neurons in the hidden layers (default: 512)')
parser.add_argument('--num_epochs_cycles', type=int, default=1, metavar='N')
parser.add_argument('--num_rollout_steps', type=int, default=180, metavar='N')
parser.add_argument('--number_of_train_steps', type=int, default=90, metavar='N')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N', help='size of replay buffer (default: 1000000)')
parser.add_argument('--train_step', default=2000)
parser.add_argument('--save_interval', default=100, help='save model interval')
parser.add_argument('--print_interval', default=10)
parser.add_argument('--seed', default=0)
parser.add_argument('--mode', default="train", help='train')
parser.add_argument('--beta', default=0.2)
parser.add_argument('--objective_num', default=2)
parser.add_argument('--user_preference', default=None, help='None or degree, for example, 60,30')
args = parser.parse_args()

# Set environment
env = gym.make(args.env_name)

# Set a random seed
env.seed(args.seed)
np.random.seed(args.seed)
rn.seed(args.seed)
torch.manual_seed(args.seed)

args.ou_noise = not args.no_ou_noise
ounoise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]),
                                       sigma=float(args.noise_scale) * np.ones(env.action_space.shape[0])
                                       ) if args.ou_noise else None
param_noise = AdaptiveParamNoiseSpec(initial_stddev=args.noise_scale,
                                     desired_action_stddev=args.noise_scale) if args.param_noise else None


model_dir_base = os.getcwd() + '/models/' + args.env_name
train_result_dir = os.getcwd() + '/results/train/' + args.env_name

if not os.path.exists(os.getcwd() + '/models/'):
    os.makedirs(os.getcwd() + '/models/')

if not os.path.exists(train_result_dir):
    os.makedirs(train_result_dir)


def reset_noise(a, a_noise, p_noise):
    if a_noise is not None:
        a_noise.reset()
    if p_noise is not None:
        a.perturb_actor_parameters(param_noise)


class Train:
    def __init__(self):
        print("Env_Name------>", args.env_name)
        print("Obj_Dimension------>", env.obj_dim)
        print("Obs_Dimension------>", env.observation_space.shape[0])

        self.model_v = cmo_ddpg.DDPG(gamma=args.gamma, tau=args.tau, hidden_size=args.hidden_size, num_inputs=env.observation_space.shape[0],
                 action_space=env.action_space, reward_space=args.objective_num, train_mode=True, replay_size=args.replay_size, beta=args.beta)

        if args.mode == "train":
            self.model_v.train()

        self.index_ = []
        self.hypervolume_ = []
        self.E_ = []
        self.D_ = []
        self.train_steps = 0
        self.point = (0, 0)

    def black_box_function(self, w):
        s = env.reset()
        reset_noise(self.model_v, ounoise, param_noise)
        preference_wb = torch.tensor([w, 1-w])
        obj_list = []

        for t in range(args.num_epochs_cycles):
            with torch.no_grad():
                for t_rollout in range(args.num_rollout_steps):
                    if args.user_preference is not None:
                        angle = args.user_preference * math.pi/180
                        preference = torch.tensor([math.tan(angle) / (1 + math.tan(angle)), 1 / (1 + math.tan(angle))])
                        preference = self.model_v.Tensor(preference).unsqueeze(0)
                    else:  # random pick a preference if it is not specified
                        if t_rollout >= int(args.num_rollout_steps/2):
                            preference = preference_wb
                            preference = self.model_v.Tensor(preference).unsqueeze(0)
                        else:
                            if t_rollout % 10 == 0:
                                angle = rn.uniform(t_rollout/10 * math.pi / 18, (t_rollout/10 + 1) * math.pi / 18)
                                preference = torch.tensor([math.tan(angle) / (1 + math.tan(angle)), 1 / (1 + math.tan(angle))])
                                preference = self.model_v.Tensor(preference).unsqueeze(0)
                                obj = np.zeros(args.objective_num)

                    state = self.model_v.Tensor([s])
                    action = self.model_v.select_action(state, preference, ounoise, param_noise)
                    next_state_, _, done, reward_ = env.step(action.cpu().numpy()[0])

                    # test hv
                    if t_rollout < int(args.num_rollout_steps/2) and t+1 == args.num_epochs_cycles:
                        obj += reward_['obj']

                    if t_rollout < int(args.num_rollout_steps/2) and (t_rollout+1) % 10 == 0 and t+1 == args.num_epochs_cycles:
                        obj_list.append(obj.tolist())

                    mask = self.model_v.Tensor([not done])
                    next_state = self.model_v.Tensor([next_state_])
                    reward = self.model_v.Tensor([reward_['obj']])

                    self.model_v.store_transition(state, preference, action, mask, next_state, reward)

                    s = next_state_
                    if done:
                        s = env.reset()
                        reset_noise(self.model_v, ounoise, param_noise)

            if args.mode == "train" and len(self.model_v.memory) > args.batch_size:
                for t_train in range(args.number_of_train_steps):
                    if self.train_steps % args.param_noise_interval == 0 and args.param_noise:
                        episode_transitions = self.model_v.memory.sample(args.batch_size)
                        states = torch.stack([transition[0] for transition in episode_transitions], dim=0)
                        preferences = torch.stack([transition[1] for transition in episode_transitions], dim=0)
                        unperturbed_actions = self.model_v.select_action(states, preferences, None, None)
                        perturbed_actions = torch.stack([transition[2] for transition in episode_transitions], 0)
                        ddpg_dist = ddpg_distance_metric(perturbed_actions.cpu().numpy(), unperturbed_actions.cpu().numpy())
                        param_noise.adapt(ddpg_dist)

                    _ = self.model_v.update_parameters(w_bayes=preference_wb, batch_size=args.batch_size)
                    self.train_steps += 1

        obj_np = np.array(obj_list)
        index, hypervolume, E, D = compute_index(obj_np, self.point)
        self.index_.append(index)
        self.hypervolume_.append(hypervolume)
        self.E_.append(E)
        self.D_.append(D)
        print("obj_np:", obj_np)
        return index

    def learning(self):
        for n_epi in range(args.train_step):
            probe_para = rn.uniform(0, 1)
            index = self.black_box_function(probe_para)
            print("n_epi:", n_epi, "probe_para:", probe_para)

            if args.mode == "train" and (n_epi+1) > args.train_step/2 and (n_epi+1) % args.save_interval == 0:
                model_dir = model_dir_base + '-%d' % int(index) + '-%d' % (n_epi+1)

                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)

                save_model(actor=self.model_v.actor, basedir=model_dir, obs_rms=self.model_v.obs_rms, rew_rms=self.model_v.ret_rms)
                print("The models are saved!")

        df_index = pd.DataFrame([])
        df_index["index"] = self.index_
        df_index["hypervolume"] = self.hypervolume_
        df_index["E"] = self.E_
        df_index["D"] = self.D_

        if args.mode == "train":
            df_index.to_csv(train_result_dir + '/train_data.csv', index=0)

        print("self.index_mean:", np.mean(self.index_))
        print("self.hypervolume_mean:", np.mean(self.hypervolume_))
        print("self.E_mean:", np.mean(self.E_))
        print("self.D_mean:", np.mean(self.D_))

        plt.plot(self.index_, "*")
        plt.xlabel('Step')
        plt.ylabel('Index')
        plt.show()

        plt.plot(self.hypervolume_)
        plt.xlabel('Step')
        plt.ylabel('Hypervolume')
        plt.show()

        plt.plot(self.E_)
        plt.xlabel('Step')
        plt.ylabel('Evenness')
        plt.show()

        plt.plot(self.D_)
        plt.xlabel('Step')
        plt.ylabel('Density')
        plt.show()

        env.close()


if __name__ == "__main__":
    Train().learning()