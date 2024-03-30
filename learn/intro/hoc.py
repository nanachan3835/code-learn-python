import gymnasium as gym

env = gym.make("MountainCar-v0",render_mode="rgb_array")
print(env.action_space.n)

env.reset()

done = False
while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)
    print(reward, new_state)