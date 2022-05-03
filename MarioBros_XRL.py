import pickle
# import time
# from MarioBros import Mario, MarioNet, prepare_env


if __name__ == '__main__':
    with open("../frame_img.pkl", "rb") as frames_in:
        frames_images = pickle.load(frames_in)