import gymnasium as gym

class PrecisionLandingReward(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)

    def step(self,action):
        #execute the action using the original physics
        obs,reward,terminated,truncated,info = self.env.step(action)

        sim = self.unwrapped

        # check whether the lander safely came to rest
        safe_rest = (
            terminated
            and not truncated
            and not sim.game_over
            and not sim.lander.awake
        )

        if safe_rest:
            # check its horizontal pisition
            on_pad = (
                sim.helipad_x1
                <= sim.lander.position.x
                <=sim.helipad_x2
            )
            if on_pad:
                reward += 50
            else:
                reward -= 50

        return obs,reward,terminated,truncated,info


class PotentialBasedReward(gym.Wrapper):
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self.previous_potential = None

    def potential(self, observation):
        x = float(observation[0])
        vx = float(observation[2])

        return -20.0 * abs(x) - 10.0 * abs(vx)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.previous_potential = self.potential(obs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # Terminal potential is zero, including episode cutoffs.
        if terminated or truncated:
            next_potential = 0.0
        else:
            next_potential = self.potential(obs)

        shaping = (
            self.gamma * next_potential
            - self.previous_potential
        )

        reward += shaping
        self.previous_potential = next_potential

        return obs, reward, terminated, truncated, info
