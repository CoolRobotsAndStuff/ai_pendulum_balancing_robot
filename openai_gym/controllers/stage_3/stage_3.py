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
import os

from matplotlib.pyplot import step
from controller import Supervisor

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from simple_pid import PID

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
    def __init__(self, max_episode_steps=1):
        super().__init__()

        # Open AI Gym generic
        self.theta_threshold_radians = 0.2
        self.x_threshold = 0.3
        high = np.array(
            [
                np.finfo(np.float32).max,

            ],
            dtype=np.float32
        )

        actHigh = np.array(
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1
                ], 
                dtype=np.float32)
        actLow = np.array(
                [
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1
                ], 
                dtype=np.float32)
        
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

        self.best = float("-inf")
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


        # Open AI Gym generic
        return np.array([0]).astype(np.float32)

    def step(self, action):

        step_n = 0
        pid_pend = PID(action[0] * 100, action[1] * 100, action[2] * 100, setpoint=0)
        pid_pend.sample_time = 0

        pid_pos = PID(action[3] * 100, action[4] * 100, action[4] * 100, setpoint=0)

        print("Action: ", action)

        comp_pendulum_sensor = 0
        total_endpoint_velocity = 0

        reward = 0

        while True:
            super().step(self.__timestep)

            robot = self.getSelf()
            endpoint = self.getFromDef("POLE_ENDPOINT")

            robot_x_pos = robot.getPosition()[2]
            robot_x_vel = robot.getVelocity()[2]
            pendulum_sensor_val = self.__pendulum_sensor.getValue()
            endpoint_x_vel = endpoint.getVelocity()[3]
            endpoint_x_pos = endpoint.getPosition()[0]
            #print("endpoint pos:", endpoint_pos)

            comp_pendulum_sensor += pendulum_sensor_val
            if endpoint_x_vel > 0:
                total_endpoint_velocity += endpoint_x_vel
            else:
                total_endpoint_velocity -= endpoint_x_vel

            # Failed
            failed = bool(
                robot_x_pos < -self.x_threshold or
                robot_x_pos > self.x_threshold or
                pendulum_sensor_val < -self.theta_threshold_radians or
                pendulum_sensor_val > self.theta_threshold_radians
            )

            # Did it

            self.succeded = step_n >= 2000

            print("Step Number:", step_n)

            if self.succeded:
                print("SUCESS!")

            # Done

            done = bool(failed or self.succeded)
            
            difference = 0

            target = endpoint_x_pos - difference
            # Reward
            if failed:
                reward_add = -200
            else:
                #reward_add = 1
                reward_add = 1 - ((target * 4) if target > 0 else (target * -4))

            print("reward_add:", reward_add) 
            reward += reward_add
            #print("reward:", reward)

            if done:
                break
            
            step_n += 1

            # Execute the action
            for wheel in self.__wheels:
                wheel.setVelocity(pid_pend(pendulum_sensor_val) + pid_pos(target))

        # Observation
        

        self.state = np.array([target])

        print("State:", self.state)
        print("Final reward:", reward)

        self.best = max(self.best, step_n)

        print("Best so far:", self.best)


        return self.state.astype(np.float32), reward, done, {}


def main():
    # Initialize the environment
    env = OpenAIGymEnvironment()
    check_env(env)

    # Train
    #model = PPO('MlpPolicy', env, n_steps=10, verbose=1)

    model = PPO.load("stage_2.0", env=env)

    
    model.learn(total_timesteps= 10* 100)

    model.save("stage_3.0")

    # Replay
    print('Training is finished, press `Y` for replay...')
    env.wait_keyboard()

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
