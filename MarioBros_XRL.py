import pickle
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms as T
from MarioBros import Mario, MarioNet, prepare_env

if __name__ == '__main__':
    with open("frame_img.pkl", "rb") as frames_in:
        frames_images = pickle.load(frames_in)
    with open("mariosave.pkl", "rb") as f_in:
        mario = pickle.load(f_in)

    transforms = T.Compose(
        # resize image in seLf.shape dimensions and then normalize image
        [T.Resize((84, 84)), T.Normalize(0, 255)]
    )

    # THE NET TAKES AS INPUT 4 GRAY-SCALED CONSECUTIVE FRAMES STACKED TOGETHER
    # THIS VERSION TAKES AS INPUT 4 CONSECUTIVE FRAMES
    # four_frames = []
    # for i in range(4):
    #     img = frames_images[59 + i]
    #     img = np.transpose(np.asarray(img), (2, 0, 1))
    #     img = torch.tensor(img, dtype=torch.float64)
    #     bw_transform = T.Grayscale()
    #     img = bw_transform(img)
    #     img = transforms(img)
    #     four_frames.append(img)
    # state = torch.stack((four_frames[0], four_frames[1], four_frames[2], four_frames[3]))
    # state = state.squeeze(1)
    # state = state.unsqueeze(0)
    #
    # net = mario.net.double()
    # action = net(state, model="target")
    # print(action)

    mod_frames = []
    for i in range(len(frames_images)):
        img = frames_images[i]
        img = np.transpose(np.asarray(img), (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float64)
        bw_transform = T.Grayscale()
        img = bw_transform(img)
        img = transforms(img)
        mod_frames.append(img)

    act = []

    net = mario.net.double()
    for i in range(4,len(frames_images)):
        state = torch.stack((mod_frames[i - 0], mod_frames[i - 1], mod_frames[i - 2], mod_frames[i - 3]))
        state = state.squeeze(1) # 4 3 2 1
        state = state.unsqueeze(0)
        act.append(net(state, model="online"))


    labels_actions = []
    for i in range(len(act)):
        if torch.argmax(act[i], axis=1).item() == 0:
            labels_actions.append("right")
        elif torch.argmax(act[i], axis=1).item() == 1:
            labels_actions.append("right + A")
        elif torch.argmax(act[i], axis=1).item() == 2:
            labels_actions.append("right + B")
        elif torch.argmax(act[i], axis=1).item() == 3:
            labels_actions.append("right + A + B")
        else:
            labels_actions.append("A")

    # code for displaying multiple images in one figure

    # create figure
    fig = plt.figure(figsize=(20,20))

    # setting values to rows and column variables
    rows = 8
    columns = 8

    pil_im = []
    for i in range(4,len(frames_images)):
        # reading images
        pil_im.append(Image.fromarray(frames_images[i]))

    for i in range(64):
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, i + 1)

        # showing image
        plt.imshow(pil_im[i])
        plt.axis('off')
        plt.title(str(i) + ". " +labels_actions[i])

    # set the spacing between subplots
    plt.subplots_adjust(top = 0.98, bottom = 0, right = 0.99, left = 0,wspace=0, hspace=0.1)
    plt.margins(0, 0)
    fig.show()