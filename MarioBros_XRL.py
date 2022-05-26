import pickle
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms as T
from MarioBros import Mario, MarioNet, prepare_env
from skimage.segmentation import mark_boundaries
from lime import lime_image

if __name__ == '__main__':
    with open("frame_img.pkl", "rb") as frames_in:
        frames_images = pickle.load(frames_in)

    env = prepare_env()
    marionet_file = 'marionetsave.pt'
    marionet = MarioNet((4, 84, 84), env.action_space.n).float()
    marionet.load_state_dict(torch.load(marionet_file, map_location=torch.device('cpu'))['model'])
    marionet.eval()

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

    net = marionet.double()
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




    # LIME IMPLEMENTATION
    def concatenate_img(im1, im2, im3, im4, plot_flag = False):
    # CONCATENATE IMAGES (OK PER INPUT)
        vertical = np.concatenate((im4, im2), axis=0)
        vertical1 = np.concatenate((im3, im1), axis=0)
        horiz = np.concatenate((vertical, vertical1), axis=1)
        if plot_flag == True:
            Image.fromarray(horiz).show()
        return horiz

    def separate_img(image):
        # SEPARATE IMAGES
        hsep = np.hsplit(image, [256])
        hsep0 = hsep[0]
        hsep1 = hsep[1]
        vsep0 = np.vsplit(hsep0, [240])
        vsep1 = np.vsplit(hsep1, [240])
        images = [vsep0[0], vsep1[0],vsep0[1],vsep1[1]]
        return images

    def image_to_state(list_img):
        four_frames = []
        for i in range(4):
            img = list_img[i]
            img = np.resize(img, (1, 256, 512, 3))
            img = img.squeeze(0)
            img = np.transpose(np.asarray(img), (2, 0, 1))
            img = torch.tensor(img, dtype=torch.float64)
            bw_transform = T.Grayscale()
            img = bw_transform(img)
            img = transforms(img)
            four_frames.append(img)
        state_lime = torch.stack((four_frames[0], four_frames[1], four_frames[2], four_frames[3]))
        state_lime = state_lime.squeeze(1)
        state_lime = state_lime.unsqueeze(0)
        state_lime = state_lime.expand(10, -1,-1,-1)
        return state_lime

    concat_im = concatenate_img(frames_images[26], frames_images[27], frames_images[28], frames_images[29])
    def net_lime(image):
        list_images = separate_img(image)
        state_lime = image_to_state(list_images)
        return net(state_lime, model = 'online').detach().numpy()


    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(concat_im, net_lime, top_labels=5, hide_color=0, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5,
                                                hide_rest=False)

    img_boundry2 = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry2)
    plt.show()