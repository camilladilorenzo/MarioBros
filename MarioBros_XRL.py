import pickle
import PIL
import numpy as np
import torch

if __name__ == '__main__':
    with open("../frame_img.pkl", "rb") as frames_in:
        frames_images = pickle.load(frames_in)
    # def permute_orientation(observation):
    #     # permute [Height, Width, Colour] array to [Colour, Height, Width] tensor
    #     observation = np.transpose(observation, (2, 0, 1))
    #     observation = torch.tensor(observation.copy(), dtype=torch.float)
    #     return observation
    frames_tens = []
    for i in range(len(frames_images)):
        img = np.transpose(np.asarray(frames_images[i]), (2, 0, 1))
        frames_tens.append(torch.tensor(img, dtype=torch.float))

