import pickle
import time
from MarioBros import Mario, MarioNet, prepare_env


if __name__ == '__main__':
    with open("../mariosave.pkl", "rb") as f_in:
        mario = pickle.load(f_in)

    env = prepare_env()
    frames = []
    episodes = 5
    for e in range(episodes):
        print(e)
        done = False
        state = env.reset()
        while not done:
            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)
            frames.append(env.render(mode="rgb_array"))
            time.sleep(0.05)

            # Update state
            state = next_state
            # Check if end of game
            if done or info["flag_get"]:
                break

    with open("../frame_img.pkl", "wb") as frames_out:
        pickle.dump(frames, frames_out)
