import torch
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from network import Critic, Actor
import numpy as np
from utils import RunningMeanStd
from replay_memory import ReplayMemory, Transition


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def normalize(x, stats, device):
    if stats is None:
        return x
    return (x - torch.Tensor(stats.mean).to(device)) / torch.Tensor(stats.var).sqrt().to(device)


class DDPG:
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, reward_space, train_mode, replay_size, beta, normalize_obs=True, normalize_returns=False, critic_l2_reg=1e-2):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.enabled = False
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.device = torch.device('cpu')
            self.Tensor = torch.FloatTensor

        self.beta = beta
        self.train_mode = train_mode

        self.num_inputs = num_inputs
        self.action_space = action_space
        self.reward_space = reward_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space, self.reward_space).to(self.device)
        if self.train_mode:
            self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space, self.reward_space).to(self.device)
            self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space, self.reward_space).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

            self.critic = Critic(hidden_size, self.num_inputs, self.action_space, self.reward_space).to(self.device)
            self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space, self.reward_space).to(self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=1e-3, weight_decay=critic_l2_reg)

            self.target_error = 0.5
            self.dual = torch.zeros(64, requires_grad=True, device=self.device)
            self.dual_optim = Adam([self.dual], lr=1e-4)
            self.dual = self.dual.unsqueeze(1)

            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_target, self.critic)

        self.gamma = gamma
        self.tau = tau
        self.normalize_observations = normalize_obs
        self.normalize_returns = normalize_returns

        if self.normalize_observations:
            self.obs_rms = RunningMeanStd(shape=num_inputs)
        else:
            self.obs_rms = None

        if self.normalize_returns:
            self.ret_rms = RunningMeanStd(shape=1)
            self.ret = 0
            self.cliprew = 10.0
        else:
            self.ret_rms = None

        self.memory = ReplayMemory(replay_size)

    def train(self):
        self.actor.train()
        if self.train_mode:
            self.critic.train()

    def select_action(self, state, preference, action_noise=None, param_noise=None, v_min=-1, v_max=1):
        state = normalize(Variable(state).to(self.device), self.obs_rms, self.device)
        preference = Variable(preference).to(self.device)

        if param_noise is not None:
            mu = self.actor_perturbed(state, preference)
        else:
            mu = self.actor(state, preference)
        mu = mu.data
        if action_noise is not None:
            mu += self.Tensor(action_noise()).to(self.device)
        mu = mu.clamp(v_min, v_max)

        return mu

    def update_model(self, state_batch, action_batch, reward_batch, preference_batch, mask_batch, next_state_batch, preference_bayes):
        # CRITIC LOSS
        next_action_actor_batch = self.actor_target(next_state_batch, preference_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_actor_batch, preference_batch)
        expected_state_action_batch = reward_batch + self.gamma * mask_batch * next_state_action_values

        self.critic_optim.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch, preference_batch)
        w_state_action_value_batch = torch.bmm(state_action_batch.unsqueeze(1), preference_batch.unsqueeze(2)).squeeze(
            1)
        w_expected_state_action_value_batch = torch.bmm(expected_state_action_batch.unsqueeze(1),
                                                        preference_batch.unsqueeze(2)).squeeze(1)

        value_loss1 = F.mse_loss(w_state_action_value_batch.view(-1), w_expected_state_action_value_batch.view(-1))
        value_loss2 = F.mse_loss(state_action_batch.view(-1), expected_state_action_batch.view(-1))

        value_loss = self.beta * value_loss1 + (1 - self.beta) * value_loss2
        value_loss.backward()
        self.critic_optim.step()

        # ACTOR LOSS
        self.actor_optim.zero_grad()
        action = self.actor(state_batch, preference_batch)
        action_b = self.actor(state_batch, preference_bayes)

        pareto_f = self.critic(state_batch, action, preference_batch)
        pareto_f_b = self.critic(state_batch, action_b, preference_bayes)

        policy_loss_sum = torch.bmm(pareto_f.unsqueeze(1), preference_batch.unsqueeze(2)).squeeze(1)
        policy_loss_sum_b = torch.bmm(pareto_f_b.unsqueeze(1), preference_bayes.unsqueeze(2)).squeeze(1)

        Q_norm2 = torch.norm(pareto_f, p=2, dim=1).unsqueeze(1)
        P_norm2 = torch.norm(preference_batch, p=2, dim=1).unsqueeze(1)
        Qn = pareto_f / Q_norm2
        Pn = preference_batch / P_norm2
        QP = Qn - Pn
        QP_norm2 = torch.norm(QP, p=2, dim=1).unsqueeze(1)

        Q_norm2_b = torch.norm(pareto_f_b, p=2, dim=1).unsqueeze(1)
        P_norm2_b = torch.norm(preference_bayes, p=2, dim=1).unsqueeze(1)
        Qn_b = pareto_f_b / Q_norm2_b
        Pn_b = preference_bayes / P_norm2_b
        QP_b = Qn_b - Pn_b
        QP_norm2_b = torch.norm(QP_b, p=2, dim=1).unsqueeze(1)

        policy_loss = -policy_loss_sum - self.dual.exp() * (self.target_error - QP_norm2 ** 2)
        policy_loss_b = -policy_loss_sum_b - self.dual.exp() * (self.target_error - QP_norm2_b ** 2)

        policy_loss = policy_loss.mean() + policy_loss_b.mean()
        dual_loss_actor = (self.dual.exp() * (self.target_error - QP_norm2 ** 2)).mean() + (
                self.dual.exp() * (self.target_error - QP_norm2_b ** 2)).mean()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        self.dual_optim.zero_grad()
        dual_loss_actor.backward(retain_graph=True)
        self.dual_optim.step()

        return pareto_f.detach().numpy()

    def store_transition(self, state, preference, action, mask, next_state, reward):
        B = state.shape[0]
        for b in range(B):
            self.memory.push(state[b], preference[b], action[b], mask[b], next_state[b], reward[b])
            if self.normalize_observations:
                self.obs_rms.update(state[b].cpu().numpy())
            if self.normalize_returns:
                self.ret = self.ret * self.gamma + reward[b]
                self.ret_rms.update(np.array([self.ret]))
                if mask[b] == 0:  # if terminal is True
                    self.ret = 0

    def update_parameters(self, w_bayes, batch_size):
        preference_bayes = w_bayes.expand(batch_size, -1)
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = normalize(Variable(torch.stack(batch.state)).to(self.device), self.obs_rms, self.device)
        action_batch = Variable(torch.stack(batch.action)).to(self.device)
        reward_batch = normalize(Variable(torch.stack(batch.reward)).to(self.device), self.ret_rms, self.device)
        preference_batch = Variable(torch.stack(batch.preference)).to(self.device)
        mask_batch = Variable(torch.stack(batch.mask)).to(self.device).unsqueeze(1)
        next_state_batch = normalize(Variable(torch.stack(batch.next_state)).to(self.device), self.obs_rms, self.device)

        if self.normalize_returns:
            reward_batch = torch.clamp(reward_batch, -self.cliprew, self.cliprew)

        pareto_f = self.update_model(state_batch, action_batch, reward_batch, preference_batch,
                                                                        mask_batch, next_state_batch, preference_bayes)

        self.soft_update()

        return pareto_f

    def soft_update(self):
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape).to(self.device) * param_noise.current_stddev
