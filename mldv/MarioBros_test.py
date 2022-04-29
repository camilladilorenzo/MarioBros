import pickle
import time

from MarioBros import Mario, MarioNet, prepare_env


if __name__ == '__main__':
    with open("../mariosave.pkl", "rb") as f_in:
        mario = pickle.load(f_in)

    env = prepare_env()

    done = False
    state = env.reset()
    while not done:
        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.05)

        # Update state
        state = next_state

    env.close()