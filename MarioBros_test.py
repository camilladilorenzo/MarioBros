import pickle
import time
from pathlib import Path
import datetime
from MarioBros import Mario, MarioNet, prepare_env, MetricLogger


if __name__ == '__main__':
    with open("../mariosave.pkl", "rb") as f_in:
        mario = pickle.load(f_in)

    save_dir = Path("../checkpoints_test") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    logger = MetricLogger(save_dir)
    env = prepare_env()
    episodes = 5
    for e in range(episodes):
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
            # Check if end of game
            if done or info["flag_get"]:
                break

            logger.log_episode()
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
        env.close()