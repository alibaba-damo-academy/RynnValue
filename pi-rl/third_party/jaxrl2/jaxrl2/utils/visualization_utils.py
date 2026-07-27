# adapted from jaxrl2

import numpy as np
import jax.numpy as jnp
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def action2img(action, res, channels, action_scale):
    assert action.size == 2  # can only plot 2-dimensional actions
    img = np.zeros((res, res, channels), dtype=np.float32).copy()
    start_pt = res / 2 * np.ones((2,))
    end_pt = start_pt + action * action_scale * (res / 2 - 1) * np.array([1, -1])  # swaps last dimension
    np2pt = lambda x: tuple(np.asarray(x, int))
    img = cv2.arrowedLine(img, np2pt(start_pt), np2pt(end_pt), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.2)
    return img

def batch_action2img(actions, res, channels, action_scale=50):
    batch = actions.shape[0]
    im = np.empty((batch, res, res, channels), dtype=np.float32)
    for b in range(batch):
        im[b] = action2img(actions[b], res, channels, action_scale)
    return im

def visualize_image_actions(images, gtruth_actions, pred_actions):
    gtruth_action_row1 = batch_action2img(gtruth_actions[:, :2], 128, 3, action_scale=3)
    gtruth_action_row1 = np.concatenate(np_unstack(gtruth_action_row1, axis=0), axis=1)
    pred_action_row1 = batch_action2img(pred_actions[:, :2], 128, 3, action_scale=3)
    pred_action_row1 = np.concatenate(np_unstack(pred_action_row1, axis=0), axis=1)
    sel_image_row = np.concatenate(np_unstack(images, axis=0), axis=1)
    image_rows = [sel_image_row, gtruth_action_row1, pred_action_row1]
    out = np.concatenate(image_rows, axis=0)
    return out


def visualize_states_rewards(states, rewards, target_point):
    states = states.squeeze()
    rewards = rewards.squeeze()

    fig, axs = plt.subplots(7, 1)
    fig.set_size_inches(5, 15)
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(states)])

    axs[0].plot(states[:, 0], linestyle='--', marker='o')
    axs[0].set_ylabel('states_x')
    axs[1].plot(states[:, 1], linestyle='--', marker='o')
    axs[1].set_ylabel('states_y')
    axs[2].plot(states[:, 2], linestyle='--', marker='o')
    axs[2].set_ylabel('states_z')

    axs[3].plot(np.abs(states[:, 0] - target_point[0]), linestyle='--', marker='o')
    axs[3].set_ylabel('norm_x')
    axs[4].plot(np.abs(states[:, 1] - target_point[1]), linestyle='--', marker='o')
    axs[4].set_ylabel('norm_y')
    axs[5].plot(np.abs(states[:, 2] - target_point[2]), linestyle='--', marker='o')
    axs[5].set_ylabel('norm_z')

    axs[6].plot(rewards, linestyle='--', marker='o')
    axs[6].set_ylabel('rewards')

    plt.tight_layout()

    canvas.draw()  # draw the canvas, cache the renderer
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return out_image

def add_text_to_images(img_list, string_list):
    from PIL import Image
    from PIL import ImageDraw
    out = []
    for im, string in zip(img_list, string_list):
        im = Image.fromarray(np.array(im).astype(np.uint8))
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), string, fill=(255, 0, 0, 128))
        out.append(np.array(im))
    return out

def sigmoid(x):
    return 1. / (1. + jnp.exp(-x))

def visualize_image_rewards(images, gtruth_rewards, pred_rewards, obs, task_id_mapping):
    id_task_mapping = {v : k for (k, v) in task_id_mapping.items()}
    sel_images = np_unstack(images, axis=0)
    sel_images = add_text_to_images(sel_images, ["{:.2f} \n{:.2f} \nTask {}".format(gtruth_rewards[i], sigmoid(pred_rewards[i, 0]), np.argmax(obs['task_id'][i])) for i in range(gtruth_rewards.shape[0])])
    sel_image_row = np.concatenate(sel_images, axis=1)
    return sel_image_row
