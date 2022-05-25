import pickle
import time, datetime
from MarioBros import prepare_env
from MarioBros import Mario, MarioNet, MetricLogger
import copy
import torch
from pathlib import Path


if __name__ == '__main__':
    env = prepare_env()

    marionet_file = 'marionetsave.pt'
    marionet = MarioNet((4, 84, 84), env.action_space.n).float()
    marionet.load_state_dict(torch.load(marionet_file, map_location=torch.device('cpu'))['model'])
    marionet.eval()

    frames = []
    episodes = 1
    for e in range(episodes):
        print(e)
        done = False
        state = env.reset()
        while not done:
            # Run agent on the state
            state = torch.tensor(state.__array__()).unsqueeze(0)
            net_evaluation = marionet(state, model="online")
            action = torch.argmax(net_evaluation).item()
            print(net_evaluation.tolist(), " -> ", action)

            frame = env.render(mode='rgb_array')
            frames.append(copy.deepcopy(frame))

            # Agent performs action
            next_state, reward, done, info = env.step(action)
            time.sleep(0.04)

            # Update state
            state = next_state
            # Check if end of game
            if done or info["flag_get"]:
                break

    with open("frame_img.pkl", "wb") as frames_out:
        pickle.dump(frames, frames_out)


