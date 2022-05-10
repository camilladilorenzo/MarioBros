import pickle
# from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
from MarioBros import Mario, MarioNet, GrayScaleObservation
# from gym.wrappers import FrameStack


if __name__ == '__main__':
    with open("frame_img.pkl", "rb") as frames_in:
        frames_images = pickle.load(frames_in)
    with open("mariosave.pkl", "rb") as f_in:
        mario = pickle.load(f_in)

    # Image.fromarray(frames_images[20]).show()
    # Image.fromarray(frames_images[50]).show()
    # Image.fromarray(frames_images[59]).show()

    transforms = T.Compose(
        # resize image in seLf.shape dimensions and then normalize image
        [T.Resize((84, 84)), T.Normalize(0, 255)]
    )
    img = frames_images[59]
    img = np.transpose(np.asarray(img), (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float64)
    bw_transform = T.Grayscale()
    img = bw_transform(img)

    #state = FrameStack(img, num_stack=4)
    state = transforms(img).unsqueeze(0)
    net = mario.net.double()
    action = net(state, model = "target")
    print(action)

