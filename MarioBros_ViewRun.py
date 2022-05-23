import pickle
import time

import torch

from MarioBros import prepare_env
from MarioBros import Mario, MarioNet
import copy

if __name__ == '__main__':
    env = prepare_env()

    marionet_file = 'marionetsave.pt'
    marionet = MarioNet((4, 84, 84), env.action_space.n).float()
    marionet.load_state_dict(torch.load(marionet_file, map_location=torch.device('cpu'))['model'])

    done = False
    state = env.reset()
    while not done:

        state = torch.tensor(state.__array__()).unsqueeze(0)
        net_evaluation = marionet(state, model="online")
        action = torch.argmax(net_evaluation).item()
        print(net_evaluation.tolist(), " -> ", action)

        env.render()
        time.sleep(0.04)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Update state
        state = next_state
        # Check if end of game
        if done or info["flag_get"]:
            break
