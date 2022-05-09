import pickle
from PIL import Image
import numpy as np
import torch
from MarioBros import Mario, MarioNet

# import torchvision
# from torchvision import datasets, transforms
# from torch import nn, optim
# from torch.nn import functional as F
# import shap

if __name__ == '__main__':
    with open("frame_img.pkl", "rb") as frames_in:
        frames_images = pickle.load(frames_in)
    with open("mariosave.pkl", "rb") as f_in:
        mario = pickle.load(f_in)

    img = frames_images[20]
    state = torch.tensor(img, dtype=torch.float)
    state = state.unsqueeze(0)
    action = mario.net(state, model = "online")
    print(action)
    Image.fromarray(frames_images[20]).show()
    Image.fromarray(frames_images[130]).show()

    # frames_tens = []
    # for i in range(len(frames_images)):
    #     img = np.transpose(np.asarray(frames_images[i]), (2, 0, 1))
    #     frames_tens.append(torch.tensor(img, dtype=torch.float))



