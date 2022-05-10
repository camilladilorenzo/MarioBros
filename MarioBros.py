# USEFUL LIBRARIES
import copy
import datetime
import random
import time
from collections import deque
from pathlib import Path

# import Gym that is an OpenAI toolkit for RL
import gym
# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.spaces import Box
from gym.wrappers import FrameStack
# NES emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace
from torch import nn
from torchvision import transforms as T
import pickle


# PREPROCESS THE ENVIRONMENT
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)  # initialize the env attribute from the base class
        self._skip = skip  # initialize skip which is a given number

    def step(self, action):
        """Repeat an action and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):  # _ takes the skip value from the initialized parameters
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        # initialize environment from the super class
        super().__init__(env)
        # extraction of the image dimensions that is [240, 256]
        obs_shape = self.observation_space.shape[:2]
        # create a box of dimensions of the image with color information
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [Height, Width, Colour] array to [Colour, Height, Width] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        # First permute orientation (let the shape contain first colour and the dimensions)
        observation = self.permute_orientation(observation)
        # Save as transform the greyscale function from the torchvision library (T is for transforms)
        transform = T.Grayscale()
        # Finally, apply the transformation to the observation
        observation = transform(observation)
        # Now observation should be in greyscale
        return observation


"""gym.ObservationWrapper: Used to modify the observations returned by the environment. To do this, override the 
observation method of the environment. This method accepts a single parameter (the observation to be modified) and 
returns the modified observation. """

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Using compose function we are creating a function containing the transformation tha we have to do on the image
        transforms = T.Compose(
            # resize image in seLf.shape dimensions and then normalize image
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        # apply function to observations
        observation = transforms(observation).squeeze(0)
        # squeeze: used when we want to remove single-dimensional entries from the shape of an array.
        return observation

# AGENT

# NEURAL NETWORK
class MarioNet(nn.Module):
    """
    mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        # init values for the function act
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        # Mario's Deep Neural  Network will be implemented in Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        # If cuda is available then set device cuda otherwise execute with cpu
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        # no. experiences between saving MarioNet
        self.curr_step = 0
        self.save_every = 5e5  # save every 500000 experiences

        # init values for the function cache and recall
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        # init values for function TD target and TD estimate
        self.gamma = 0.9

        # init values for updating model
        # first we initialize the method used to
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        # define the loss measure
        self.loss_fn = torch.nn.SmoothL1Loss()

        # init values for learning
        self.burnin = 1e4  # min. experience before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    # ACT
    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)

        Outputs:
        action_idx(int): An integer representing which action MArio will perform

        """
        # EXPLORE
        # if a uniform random number is less than epsilon
        # (epsilon = exploration rate which decreases as the number of action made increases)
        if np.random.rand() < self.exploration_rate:
            # sample a random action from the action space
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        # else (if the random number is greater than epsilon)
        else:
            # create an array with all possible state
            state = state.__array__()
            # convert this array into a tensor
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            # use the net to apply an action value to each state
            action_values = self.net(state, model="online")
            # select the action which has the greatest value in the Q-matrix
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    # CACHE AND RECALL
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to memory (replay buffer)

        Inputs:
            -- state (LazyFrame)
            -- next state (LazyFrame)
            -- action (int)
            -- reward (float)
            -- done (bool)
        """
        # First convert the input state and next state into array
        state = state.__array__()
        next_state = next_state.__array__()

        # After that if cuda is available use it otherwise does not use it
        # and transform all inputs into tensors
        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        # Finally, append all to the memory deque
        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        # sample from memory
        batch = random.sample(self.memory, self.batch_size)
        # Extract from the tuples sampled the features (state, next_state, action, reward, done)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    # TD ESTIMATE AND TD TARGET
    def td_estimate(self, state, action):
        # Q_online and Q_target are two ConNets
        # since model = "online" we will apply Q_online
        # over a given state an actions
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s, a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # We compute Q_online in order to find a'
        next_state_Q = self.net(next_state, model="online")
        # since we don't know a' we compute it as the action which
        # maximizes Q_online
        best_action = torch.argmax(next_state_Q, axis=1)
        # Having a' computed as above we can apply the second ConvNet
        # Q_target(s', a')
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        # at the end we return reward + gamma*Q_target(a', s')
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    # UPDATE THE MODEL
    def update_Q_online(self, td_estimate, td_target):
        # Evaluate the loss
        loss = self.loss_fn(td_estimate, td_target)
        # Sets the gradients of all optimized torch.Tensor s to zero.
        self.optimizer.zero_grad()
        # Computes the gradient of current tensor w.r.t. graph leaves
        loss.backward()
        # the function step is used to update parameters
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        # Loads a model’s parameter dictionary using a deserialized state_dict
        self.net.target.load_state_dict(self.net.online.state_dict())


    # SAVE THE MODEL
    def save(self):
        save_path = (
                self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )

        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    # LEARN
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD estimate
        td_est = self.td_estimate(state, action)
        # Get TD target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagation loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


# LOGGING
class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"

        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        """Mark end of episode"""
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)

        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()


# PREPARE THE ENVIRONMENT AND TRAIN THE MODEL
def prepare_env():
    # INITIALIZE THE ENVIRONMENT

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    # The action space of the game comprehend the 12 following moves:
    # - NOOP --> do nothing
    # - right --> walk right
    # - right + A --> jump right
    # - right + B --> run right
    # - right + A + B --> run and jump right
    # - A --> jump
    # - left --> walk left
    # - left + A --> jump left
    # - left + B --> run left
    # - left + A + B --> run and jump left
    # - down -->  duck, enter a pipe or climb downwards on a beanstalk
    # - up -->  climb upwards on a beanstalk

    env = JoypadSpace(env, [["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"],
                            ["A"]])
    # , ["left"], ["left", "A"], ["left", "B"], ["left", "A", "B"]
    env.reset()
    next_state, reward, done, info = env.step(action=0)
    print(f"{next_state.shape}, \n {reward}, \n {done}, \n {info}")

    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    return env


if __name__ == '__main__':
    env = prepare_env()
    # train the model
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("../checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 70
    for e in range(episodes):
        print(e)
        state = env.reset()

        # Play the game!
        while True:
            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)
            env.render()

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
        # if e % 20 == 0:
        # logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    env.close()

    # serialize pickle
    with open("mariosave.pkl", "wb") as f_out:
        pickle.dump(mario, f_out)