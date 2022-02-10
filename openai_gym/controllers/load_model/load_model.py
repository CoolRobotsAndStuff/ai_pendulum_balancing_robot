# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time
from controller import Supervisor

try:
    import gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym stable_baselines3"'
    )


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()

        # Open AI Gym generic
        self.theta_threshold_radians = 0.2
        self.x_threshold = 0.3
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max
            ],
            dtype=np.float32
        )

        actHigh = np.array([1], dtype=np.float32)
        actLow = np.array([-1], dtype=np.float32)
        #self.action_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Box(actLow, actHigh, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__wheels = []
        self.__pendulum_sensor = None

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

        self.step_n = 0
        self.succeded = False
        #self.target = 0
        #self.target_dir = 0

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Motors
        self.__wheels = []
        for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            self.__wheels.append(wheel)

        # Sensors
        self.__pendulum_sensor = self.getDevice('position sensor')
        self.__pendulum_sensor.enable(self.__timestep)

        # Internals
        super().step(self.__timestep)

        #self.target = 0
        #self.target_dir = 0
        self.step_n = 0

        # Open AI Gym generic
        return np.array([0, 0, 0, 0]).astype(np.float32)

    def step(self, action):
        # Execute the action
        for wheel in self.__wheels:
            wheel.setVelocity(action[0] * 5)
        super().step(self.__timestep)

        # Observation
        robot = self.getSelf()
        endpoint = self.getFromDef("POLE_ENDPOINT")
        self.state = np.array([robot.getPosition()[2], robot.getVelocity()[2],
                               self.__pendulum_sensor.getValue(), endpoint.getVelocity()[3]])

        # Failed
        failed = bool(
            self.state[0] < -self.x_threshold or
            self.state[0] > self.x_threshold or
            self.state[2] < -self.theta_threshold_radians or
            self.state[2] > self.theta_threshold_radians
        )

        # Did it

        self.succeded = self.step_n >= 999

        if self.succeded:
            print("SUCESS!")

        # Done

        done = bool(failed or self.succeded)
        
        
        endpoint_x = endpoint.getPosition()[0]
        #print("endpoint pos:", endpoint_pos)

        # Reward
        if failed: 
            reward = 0 
        else:
            reward = 1 - min(((endpoint_x * 2) if endpoint_x > 0 else (endpoint_x * -2)), 1)
            print("reward:", reward)
        
        try:
            assert 1 >= reward >= 0
        
        except AssertionError:
            print("reward:", reward)


        self.step_n += 1

        """
        if self.target_dir == 0:
            self.target += 0.001
        
        else:
            self.target -= 0.001
        
        if self.target > 0.3:
            self.target_dir = 1
        elif self.target < -0.3:
            self.target_dir = 0

        print("target: ", self.target)
        """

        return self.state.astype(np.float32), reward, done, {}


def main():
    # Initialize the environment
    env = OpenAIGymEnvironment()
    check_env(env)

    # Train
    model = PPO.load("model_v1")

    #model.save("model_v1")

    # Replay
    print('Training is finished, press `Y` for replay...')
    env.wait_keyboard()

    obs = env.reset()
    for _ in range(2048 * 1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
