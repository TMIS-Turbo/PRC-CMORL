# Swimmer-v3 env
# 3 objectives

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 3
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/swimmer.xml"), frame_skip = 4)

    def step(self, a):
        ctrl_cost_coeff = 0.1

        a = np.clip(a, -1, 1)
        self.do_simulation(a, self.frame_skip)

        reward_fwd_x = max(0, self.sim.data.qvel[0])/2
        reward_fwd_y = max(0, self.sim.data.qvel[1])/2
        reward_ctrl = max(0, 1 - (reward_fwd_x**2+reward_fwd_y**2)**0.5 - ctrl_cost_coeff * np.square(a).sum())
        ob = self._get_obs()

        return ob, 0., False, {'obj': np.array([reward_fwd_x, reward_fwd_y, reward_ctrl])}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()
