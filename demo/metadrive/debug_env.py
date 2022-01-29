import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE
from core.envs.md_hrl_env import MetaDriveHRLEnv
def get_waypoint_list():
    x = np.arange(0, 6.2, 0.2)
    LENGTH = 0 # 4.51
    y = 1 * np.cos(np.pi*2 / 6.0 * x)-1
    x = x + LENGTH/2
    lst = []
    for i in range(x.shape[0]):
        lst.append([x[i],y[i]])
    return lst
def get_waypoint_list1():
    x = np.arange(0, 6.2, 0.2)
    LENGTH = 0 # 4.51
    y = 1 * np.cos(np.pi*2 / 6.0 * x)-1
    y = -y
    x = x + LENGTH/2
    lst = []
    for i in range(x.shape[0]):
        lst.append([x[i],y[i]])
    return lst

def draw_multi_channels_top_down_observation(obs, show_time=4):
    num_channels = obs.shape[-1]
    assert num_channels == 5
    channel_names = [
        "Road and navigation", "Ego now and previous pos", "Neighbor at step t", "Neighbor at step t-1",
        "Neighbor at step t-2"
    ]
    fig, axs = plt.subplots(1, num_channels, figsize=(15, 4), dpi=80)
    count = 0

    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    timer = fig.canvas.new_timer(
        interval=show_time * 1000
    )  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    for i, name in enumerate(channel_names):
        count += 1
        ax = axs[i]
        ax.imshow(obs[..., i], cmap="bone")
        #print('channel_{}.min is: {} and max is: {}'.format(i, obs[..., i].min(), obs[..., i].max()))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        # print("Drawing {}-th semantic map!".format(count))
    fig.suptitle("Multi-channels Top-down Observation")
    timer.start()
    plt.show()

if __name__ == "__main__":
    config = dict(
        use_render=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="birdview", choices=["lidar", "rgb_camera", "birdview"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(dict(offscreen_render=True))
    env = MetaDriveHRLEnv(config)
    try:
        o = env.reset()
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("The observation is an numpy array with shape: ", o.shape)
            i = 0
        reward_total = 0
        
        for j in range(1, 1000000000):
            lst = get_waypoint_list()
            i += 1
            # print(env.action_type.actions_indexes["LANE_LEFT"])

            
            # if i < 10:
            #     action_zt = env.action_type.actions_indexes["Holdon"]
            if i %2 == 0:
                action = env.action_type.actions_indexes["IDLE"]
                lst = get_waypoint_list()
            # elif (i+1) % 4 == 0:
            #     action = env.action_type.actions_indexes["LANE_LEFT"]
            # else:
            #     action = env.action_type.actions_indexes["LANE_RIGHT"]
            else:
                action = env.action_type.actions_indexes["IDLE"]
                lst = get_waypoint_list1()
            action = lst

            #action_zt = env.action_type.actions_indexes["LANE_LEFT"] if i % 2 ==0 else env.action_type.actions_indexes["LANE_RIGHT"]
            o, r, d, info = env.step(action)
            reward_total += r
            #env.render(mode="top_down", film_size=(800, 800))
            #draw_multi_channels_top_down_observation(o, show_time=2)

            print('reward is: {}; and total_reward is : {}'.format(r, reward_total))
            if d or info["arrive_dest"]:
                env.reset()
                reward_total = 0
                print('reset')
                i = 0
    except:
        pass
    finally:
        env.close()
