from gym.envs.registration import register

register(
    id='my-env-v0',
    entry_point='custom_env.gym_my_env.envs:MyEnv',
)
