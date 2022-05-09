import pickle
import time
from MarioBros import Mario, MarioNet, prepare_env
import copy
from PIL import Image
import PIL.ImageDraw as ImageDraw
import numpy as np


# def _label_with_episode_number(frame, episode_num):
#     im = Image.fromarray(frame)
#     drawer = ImageDraw.Draw(im)
#     if np.mean(im) < 128:
#         text_color = (255,255,255)
#     else:
#         text_color = (0,0,0)
#     drawer.text((im.size[0]/20, im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)
#
#     return im

if __name__ == '__main__':
    with open("mariosave.pkl", "rb") as f_in:
        mario = pickle.load(f_in)

    env = prepare_env()
    frames = []
    episodes = 1
    for e in range(episodes):
        print(e)
        done = False
        state = env.reset()
        while not done:
            # Run agent on the state
            action = mario.act(state)
            frame = env.render(mode='rgb_array')
            frames.append(copy.deepcopy(frame))

            # Agent performs action
            next_state, reward, done, info = env.step(action)
            time.sleep(0.05)

            # Update state
            state = next_state
            # Check if end of game
            if done or info["flag_get"]:
                break

    with open("frame_img.pkl", "wb") as frames_out:
        pickle.dump(frames, frames_out)
