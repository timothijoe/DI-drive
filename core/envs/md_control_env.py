import os
import copy
import time
import gym
import numpy as np
from gym import spaces
from collections import defaultdict
from typing import Union, Dict, AnyStr, Tuple, Optional
from gym.envs.registration import register
import logging

from core.utils.simulator_utils.md_utils.discrete_policy import DiscreteMetaAction
from core.utils.simulator_utils.md_utils.agent_manager_utils import MacroAgentManager
from core.utils.simulator_utils.md_utils.engine_utils import initialize_engine, close_engine, \
    engine_initialized, set_global_random_seed, MacroBaseEngine
from core.utils.simulator_utils.md_utils.traffic_manager_utils import TrafficMode
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT, REPLAY_DONE
from metadrive.envs.base_env import BaseEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
# from metadrive.manager.traffic_manager import TrafficMode
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils import Config, merge_dicts, get_np_random, clip
from metadrive.utils import Config, merge_dicts, get_np_random, concat_step_infos
from metadrive.envs.base_env import BASE_DEFAULT_CONFIG
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.utils.utils import auto_termination
#from core.policy.ad_policy.traj_vae import VaeDecoder
from core.policy.ad_policy.traj_vae import WpDecoder
import torch

DIDRIVE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    use_render=False,
    environment_num=1,

    # ===== Map Config =====
    map='SSSSSSSSSS',  # int or string: an easy way to fill map_config
    random_lane_width=True,
    random_lane_num=True,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
        BaseMap.GENERATE_CONFIG: 'SSSSSSSSSS',  # None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 3,
        "exit_length": 70,
    },

    # ===== Traffic =====
    traffic_density=0.0,
    on_screen=False,
    rgb_clip=True,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Synch,  # "Respawn", "Trigger"
    random_traffic=True,  # Traffic is randomized at default.
    # this will update the vehicle_config and set to traffic
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),

    # ===== Object =====
    accident_prob=0.,  # accident may happen on each block with this probability, except multi-exits block

    # ===== Others =====
    use_AI_protector=False,
    save_level=0.5,
    is_multi_agent=False,
    vehicle_config=dict(spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)),

    # ===== Agent =====
    target_vehicle_configs={
        DEFAULT_AGENT: dict(use_special_color=True, spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 2))
    },

    # ===== Reward Scheme =====
    # See: https://github.com/decisionforce/metadrive/issues/283
    success_reward=10.0,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=1.0,
    crash_object_penalty=5.0,
    run_out_of_time_penalty = 5.0,
    driving_reward=1.0,
    speed_reward=0.5,
    use_lateral=True,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=True,
    physics_world_step_size=1e-1,

    # ===== Trajectory length =====
    seq_traj_len = 10,
    show_seq_traj = False,
    episode_max_step = 5000,

    use_jerk_penalty = False,
    use_lateral_penalty = False,


)


class MetaDriveControlEnv(BaseEnv):

    @classmethod
    def default_config(cls) -> "Config":
        #config = super(SimpleMetaDriveEnv, cls).default_config()
        config = Config(BASE_DEFAULT_CONFIG)
        config.update(DIDRIVE_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: dict = None):
        merged_config = self._merge_extra_config(config)
        global_config = self._post_process_config(merged_config)
        self.config = global_config

        # agent check
        self.num_agents = self.config["num_agents"]
        self.is_multi_agent = self.config["is_multi_agent"]
        if not self.is_multi_agent:
            assert self.num_agents == 1
        assert isinstance(self.num_agents, int) and (self.num_agents > 0 or self.num_agents == -1)

        # observation and action space
        self.agent_manager = MacroAgentManager(
            init_observations=self._get_observations(), init_action_space=self._get_action_space()
        )
        self.action_type = DiscreteMetaAction()
        #self.action_space = self.action_type.space()

        # lazy initialization, create the main vehicle in the lazy_init() func
        self.engine: Optional[MacroBaseEngine] = None
        self._top_down_renderer = None
        self.episode_steps = 0
        # self.current_seed = None

        # In MARL envs with respawn mechanism, varying episode lengths might happen.
        self.dones = None
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        self.start_seed = self.config["start_seed"]
        self.env_num = self.config["environment_num"]

        self.time = 0
        self.step_num = 0
        self.episode_rwd = 0
        self.vae_decoder = WpDecoder(
                control_num = 2,
                seq_len = 1,
                dt = 0.1
            )
        vae_load_dir = 'ckpt_files/a79_decoder_ckpt'
        #self.vae_decoder.load_state_dict(torch.load(vae_load_dir))
        self.vel_speed = 0.0

    # define a action type, and execution style
    # Now only one action will be taken, cosin function, and we set dt equals self.engine.dt
    # now that in this situation, we directly set trajectory len equals to simulation frequency

    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]):
        self.episode_steps += 1
        # if not isinstance(actions,list):
        #     action_seq = []
        #     for i in range(31):
        #         action_seq.append([actions[2 * i], actions[2 *i+1]])
        #     actions = action_seq
        #action_seq =  np.array(action_seq)
        init_state = np.zeros([1, 4])
        init_state[0,3] = self.vel_speed
        init_state = torch.from_numpy(init_state)
        #actions = np.array([1,1])
        if isinstance(actions, np.ndarray):
            batch_action = torch.from_numpy(actions)
            batch_action = torch.unsqueeze(batch_action, 0)
            batch_action = batch_action.to(torch.float32)
            init_state = init_state.to(torch.float32)
            with torch.no_grad():
                trajs = self.vae_decoder(batch_action, init_state)
            trajs = torch.cat([init_state.unsqueeze(1), trajs], dim = 1)
            trajs = trajs[:,:,:2]
            trajs = torch.squeeze(trajs, 0)
            actions = trajs.numpy()
        macro_actions = self._preprocess_macro_waypoints(actions)
        step_infos = self._step_macro_simulator(macro_actions)
        o, r, d, i = self._get_step_return(actions, step_infos)
        self.step_num = self.step_num + 1
        self.episode_rwd = self.episode_rwd + r 
        #print('step number is: {}'.format(self.step_num))
        #o = o.transpose((2,0,1))
        return o, r, d, i

    def get_waypoint_list(self):
        x = np.arange(0, 6.2, 0.2)
        LENGTH = 0 # 4.51
        y = 1 * np.cos(np.pi*2 / 6.0 * x)-1
        x = x + LENGTH/2
        lst = []
        for i in range(x.shape[0]):
            lst.append([x[i],y[i]])
        return lst

    def _merge_extra_config(self, config: Union[dict, "Config"]) -> "Config":
        config = self.default_config().update(config, allow_add_new_key=True)
        if config["vehicle_config"]["lidar"]["distance"] > 50:
            config["max_distance"] = config["vehicle_config"]["lidar"]["distance"]
        return config

    def _post_process_config(self, config):
        config = super(MetaDriveControlEnv, self)._post_process_config(config)
        if not config["rgb_clip"]:
            logging.warning(
                "You have set rgb_clip = False, which means the observation will be uint8 values in [0, 255]. "
                "Please make sure you have parsed them later before feeding them to network!"
            )
        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config()
        )
        config["vehicle_config"]["rgb_clip"] = config["rgb_clip"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        if config.get("gaussian_noise", 0) > 0:
            assert config["vehicle_config"]["lidar"]["gaussian_noise"] == 0, "You already provide config!"
            assert config["vehicle_config"]["side_detector"]["gaussian_noise"] == 0, "You already provide config!"
            assert config["vehicle_config"]["lane_line_detector"]["gaussian_noise"] == 0, "You already provide config!"
            config["vehicle_config"]["lidar"]["gaussian_noise"] = config["gaussian_noise"]
            config["vehicle_config"]["side_detector"]["gaussian_noise"] = config["gaussian_noise"]
            config["vehicle_config"]["lane_line_detector"]["gaussian_noise"] = config["gaussian_noise"]
        if config.get("dropout_prob", 0) > 0:
            assert config["vehicle_config"]["lidar"]["dropout_prob"] == 0, "You already provide config!"
            assert config["vehicle_config"]["side_detector"]["dropout_prob"] == 0, "You already provide config!"
            assert config["vehicle_config"]["lane_line_detector"]["dropout_prob"] == 0, "You already provide config!"
            config["vehicle_config"]["lidar"]["dropout_prob"] = config["dropout_prob"]
            config["vehicle_config"]["side_detector"]["dropout_prob"] = config["dropout_prob"]
            config["vehicle_config"]["lane_line_detector"]["dropout_prob"] = config["dropout_prob"]
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["target_vehicle_configs"][DEFAULT_AGENT])
            config["target_vehicle_configs"][DEFAULT_AGENT] = target_v_config
        return config

    def _get_observations(self):
        return {DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

    def done_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        done = False
        done_info = dict(
            crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False
        )
        if vehicle.arrive_destination:
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info[TerminationState.SUCCESS] = True
        elif hasattr(vehicle, 'macro_succ') and vehicle.macro_succ:
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info[TerminationState.SUCCESS] = True
        elif hasattr(vehicle, 'macro_crash') and vehicle.macro_crash:
            done = True
            logging.info("Episode ended! Reason: crash vehicle ")
            done_info[TerminationState.CRASH_VEHICLE] = True
        if self._is_out_of_road(vehicle):
            done = True
            logging.info("Episode ended! Reason: out_of_road.")
            done_info[TerminationState.OUT_OF_ROAD] = True
        if vehicle.crash_vehicle:
            done = True
            logging.info("Episode ended! Reason: crash vehicle ")
            done_info[TerminationState.CRASH_VEHICLE] = True
        if vehicle.crash_object:
            done = True
            done_info[TerminationState.CRASH_OBJECT] = True
            logging.info("Episode ended! Reason: crash object ")
        if vehicle.crash_building:
            done = True
            done_info[TerminationState.CRASH_BUILDING] = True
            logging.info("Episode ended! Reason: crash building ")
        if self.step_num >= self.config["episode_max_step"]:
            done = True
            done_info[TerminationState.CRASH_BUILDING] = True
            logging.info("Episode ended! Reason: crash building ")

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING]
        )

        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        elif self.step_num > self.config["episode_max_step"]:
            step_info['cost'] = 1
        return step_info['cost'], step_info

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        ret = vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or \
              (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.vehicles[vehicle_id]
        
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_macro_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        vehicle_heading_theta = vehicle.heading_theta
        road_heading_theta = current_lane.heading
        theta_error = self.wrap_angle(vehicle_heading_theta - road_heading_theta)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral"]:
            lateral_factor = clip(1 - 0.5 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
            #lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0
        longitude_factor = 0.2
        heading_factor = 0.15
        
        #heading_theta_rwd = 5 - 2 * np.abs(theta_error) 

        reward = 0.0
        max_spd = 10
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor *  longitude_factor * positive_road 
        reward += self.config["speed_reward"] * (vehicle.last_spd / max_spd) * positive_road * 0.1
        if vehicle.last_spd<4:
            reward -= 0.04
        reward += heading_factor * ( 0 - np.abs(theta_error))
        
        
        if self.config["use_jerk_penalty"]:
            jerk_value = self.compute_jerk_penalty(vehicle)
            reward += (0.04-jerk_value / 200.0) 
        if self.config["use_lateral_penalty"]:
            lateral_penalty = self.compute_lateral_penalty(vehicle, current_lane)
            reward -= lateral_penalty /4.0 * 0.3
        step_info["step_reward"] = reward


        if vehicle.arrive_destination:
            reward = +self.config["success_reward"]
        elif vehicle.macro_succ:
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.macro_crash:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif self.step_num >= self.config["episode_max_step"]:
            reward = - self.config["run_out_of_time_penalty"]
        return reward, step_info

    def compute_jerk_penalty(self, vehicle):
        jerk_list = []
        #vehicle = self.vehicles[vehicle_id]
        v_t0 = vehicle.penultimate_state['speed']
        theta_t0 = vehicle.penultimate_state['yaw']
        v_t1 = vehicle.traj_wp_list[0]['speed']
        theta_t1 = vehicle.traj_wp_list[0]['yaw']
        v_t2 = vehicle.traj_wp_list[1]['speed']
        theta_t2 = vehicle.traj_wp_list[1]['yaw']
        t_inverse = 1.0 / self.config['physics_world_step_size']
        first_point_jerk_x = (v_t2* np.cos(theta_t2) - 2 * v_t1 * np.cos(theta_t1) +  v_t0 * np.cos(theta_t0)) * t_inverse * t_inverse
        first_point_jerk_y = (v_t2* np.sin(theta_t2) - 2 * v_t1 * np.sin(theta_t1) +  v_t0 * np.sin(theta_t0)) * t_inverse * t_inverse
        jerk_list.append(np.array([first_point_jerk_x, first_point_jerk_y]))
        # plus one because we store the current value as first, which means the whole trajectory is seq_traj_len + 1
        for i in range(2, self.config['seq_traj_len'] + 1):
            v_t0 = vehicle.traj_wp_list[i-2]['speed']
            theta_t0 = vehicle.traj_wp_list[i-2]['yaw']
            v_t1 = vehicle.traj_wp_list[i-1]['speed']
            theta_t1 = vehicle.traj_wp_list[i-1]['yaw']
            v_t2 = vehicle.traj_wp_list[i]['speed']
            theta_t2 = vehicle.traj_wp_list[i]['yaw']    
            point_jerk_x = (v_t2* np.cos(theta_t2) - 2 * v_t1 * np.cos(theta_t1) + v_t0 * np.cos(theta_t0)) * t_inverse * t_inverse
            point_jerk_y = (v_t2* np.sin(theta_t2) - 2 * v_t1 * np.sin(theta_t1) + v_t0 * np.sin(theta_t0)) * t_inverse * t_inverse
            jerk_list.append(np.array([point_jerk_x, point_jerk_y]))
        final_jerk_value = 0
        for jerk in jerk_list:
            final_jerk_value += np.linalg.norm(jerk)
        return final_jerk_value

    def compute_lateral_penalty(self, vehicle, lane):
        final_lateral_value = 0
        for i in range(1, self.config['seq_traj_len'] +1):
            long_now, lateral_now = lane.local_coordinates(vehicle.traj_wp_list[i]['position'])
            final_lateral_value += np.abs(lateral_now)
        final_lateral_value /= float(self.config['seq_traj_len'])
        return final_lateral_value
            


    def switch_to_third_person_view(self) -> None:
        if self.main_camera is None:
            return
        self.main_camera.reset()
        if self.config["prefer_track_agent"] is not None and self.config["prefer_track_agent"] in self.vehicles.keys():
            new_v = self.vehicles[self.config["prefer_track_agent"]]
            current_track_vehicle = new_v
        else:
            if self.main_camera.is_bird_view_camera():
                current_track_vehicle = self.current_track_vehicle
            else:
                vehicles = list(self.engine.agents.values())
                if len(vehicles) <= 1:
                    return
                if self.current_track_vehicle in vehicles:
                    vehicles.remove(self.current_track_vehicle)
                new_v = get_np_random().choice(vehicles)
                current_track_vehicle = new_v
        self.main_camera.track(current_track_vehicle)
        return

    def switch_to_top_down_view(self):
        self.main_camera.stop_track()

    def _get_step_return(self, actions, engine_info):
        # update obs, dones, rewards, costs, calculate done at first !
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        rewards = {}
        for v_id, v in self.vehicles.items():
            o = self.observations[v_id].observe(v)
            # o[0,0,1] = 0
            # o[0,1,1] = 0
            # o[0,2,1] = 0
            # o[0,3,1] = v.last_spd
            self.vel_speed = v.last_spd
            # append the six-th 
            #v_o = np.ones([200, 200, 1]) * v.last_spd * 0.01
            #o = np.concatenate((o, v_o), axis = 2)
            # o_dict = {}
            # o_dict['birdview'] = o 
            # o_dict['speed'] = v.last_spd
            obses[v_id] =  o #o_dict
            done_function_result, done_infos[v_id] = self.done_function(v_id)
            rewards[v_id], reward_infos[v_id] = self.reward_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)
            done = done_function_result or self.dones[v_id]
            self.dones[v_id] = done

        should_done = engine_info.get(REPLAY_DONE, False
                                      ) or (self.config["horizon"] and self.episode_steps >= self.config["horizon"])
        termination_infos = self.for_each_vehicle(auto_termination, should_done)

        step_infos = concat_step_infos([
            engine_info,
            done_infos,
            reward_infos,
            cost_infos,
            termination_infos,
        ])

        if should_done:
            for k in self.dones:
                self.dones[k] = True

        dones = {k: self.dones[k] for k in self.vehicles.keys()}
        for v_id, r in rewards.items():
            self.episode_rewards[v_id] += r
            step_infos[v_id]["episode_reward"] = self.episode_rewards[v_id]
            self.episode_lengths[v_id] += 1
            step_infos[v_id]["episode_length"] = self.episode_lengths[v_id]
        if not self.is_multi_agent:
            return self._wrap_as_single_agent(obses), self._wrap_as_single_agent(rewards), \
                   self._wrap_as_single_agent(dones), self._wrap_as_single_agent(step_infos)
        else:
            return obses, rewards, dones, step_infos

    def setup_engine(self):
        super(MetaDriveControlEnv, self).setup_engine()
        self.engine.accept("b", self.switch_to_top_down_view)
        self.engine.accept("q", self.switch_to_third_person_view)
        from core.utils.simulator_utils.md_utils.traffic_manager_utils import MacroTrafficManager
        from core.utils.simulator_utils.md_utils.map_manager_utils import MacroMapManager
        self.engine.register_manager("map_manager", MacroMapManager())
        self.engine.register_manager("traffic_manager", MacroTrafficManager())

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else \
            get_np_random(self._DEBUG_RANDOM_SEED).randint(self.start_seed, self.start_seed + self.env_num)
        self.seed(current_seed)

    def _preprocess_macro_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray]]:
        if not self.is_multi_agent:
            # print('action.dtype: {}'.format(type(actions)))
            #print('action: {}'.format(actions))
            actions = int(actions)
            actions = {v_id: actions for v_id in self.vehicles.keys()}
        else:
            if self.config["vehicle_config"]["action_check"]:
                # Check whether some actions are not provided.
                given_keys = set(actions.keys())
                have_keys = set(self.vehicles.keys())
                assert given_keys == have_keys, "The input actions: {} have incompatible keys with existing {}!".format(
                    given_keys, have_keys
                )
            else:
                # That would be OK if extra actions is given. This is because, when evaluate a policy with naive
                # implementation, the "termination observation" will still be given in T=t-1. And at T=t, when you
                # collect action from policy(last_obs) without masking, then the action for "termination observation"
                # will still be computed. We just filter it out here.
                actions = {v_id: actions[v_id] for v_id in self.vehicles.keys()}
        return actions

    def _preprocess_macro_waypoints(self, waypoint_list: Union[np.ndarray, Dict[AnyStr, np.ndarray]]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray]]:
        if not self.is_multi_agent:
            # print('action.dtype: {}'.format(type(actions)))
            #print('action: {}'.format(actions))
            actions = waypoint_list
            actions = {v_id: actions for v_id in self.vehicles.keys()}
        return actions

    def _step_macro_simulator(self, actions):
        #simulation_frequency = 30  # 60 80
        simulation_frequency = self.config['seq_traj_len']
        policy_frequency = 1
        frames = int(simulation_frequency / policy_frequency)
        self.time = 0
        # print('seq len is: ')
        # print(self.config['seq_traj_len'])
        #print('di action pairs: {}'.format(actions))
        #actions = {vid: self.action_type.actions[vvalue] for vid, vvalue in actions.items()}
        # wp_list = self.get_waypoint_list()
        # wps = dict()
        # for vid in actions.keys():
        #     wps[vid] = wp_list
        wps = actions
        for frame in range(frames):
            # we use frame to update robot position, and use wps to represent the whole trajectory
            scene_manager_before_step_infos = self.engine.before_step_macro(frame, wps)
            self.engine.step()
            scene_manager_after_step_infos = self.engine.after_step()
        #scene_manager_after_step_infos = self.engine.after_step()
        return merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )

    # @property
    # def action_space(self) -> gym.Space:
    #     """
    #     Return observation spaces of active and controllable vehicles
    #     :return: Dict
    #     """
    #     #return self.action_type.space()
    #     return gym.spaces.Box(-50.0, 50.0, shape=(62, ), dtype=np.float32)

    def _get_reset_return(self):
        ret = {}
        self.engine.after_step()
        o = None
        print('episode reward: {}'.format(self.episode_rwd))
        self.episode_rwd = 0
        self.step_num = 0
        for v_id, v in self.vehicles.items():
            self.observations[v_id].reset(self, v)
            ret[v_id] = self.observations[v_id].observe(v)
            o = self.observations[v_id].observe(v)
            v_o = np.ones([200, 200, 1]) * v.last_spd * 0.01
            #o = np.concatenate((o, v_o), axis = 2)
            o_dict = {}
            o_dict['birdview'] = o 
            o_dict['speed'] = v.last_spd
            #obses[v_id] =  o_dict #o
            self.vel_speed = 0

            if hasattr(v, 'macro_succ'):
                v.macro_succ = False
            if hasattr(v, 'macro_crash'):
                v.macro_crash = False
            v.penultimate_state = {}
            v.penultimate_state['position'] = np.array([0,0])
            v.penultimate_state['yaw'] = 0 
            v.penultimate_state['speed'] = 0
            v.traj_wp_list = [] 
            v.traj_wp_list.append(copy.deepcopy(v.penultimate_state))
            v.traj_wp_list.append(copy.deepcopy(v.penultimate_state))
            v.last_spd = 0
        # zt_obs = zt_obs.transpose((2,0,1))
        # print('process: {}  --- > initializing: a new episode begins'.format(os.getpid()))
        self.remove_init_stop = True
        # for v_id ,v in self.vehicles.items():
        #     if hasattr(v, 'macro_succ'):
        #         p = self.engine.get_policy(v.name)
        #         target_speed = p.NORMAL_SPEED * 0.1
        #         print('target velocity: {}'.format(target_speed))
        #         v.set_velocity(v.heading, target_speed)
        if self.remove_init_stop:
            return o #o_dict
        for i in range(8):
            o, r, d, info = self.step(self.action_type.actions_indexes["Holdon"])
        for v_id ,v in self.vehicles.items():
            if hasattr(v, 'macro_succ'):
                p = self.engine.get_policy(v.name)
                target_speed = p.NORMAL_SPEED * 0.1
                #print('target velocity: {}'.format(target_speed))
                v.set_velocity(v.heading, target_speed)
        for i in range(1):
            o, r, d, info = self.step(self.action_type.actions_indexes["IDLE"])
            o = o
        return o

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # It is the true init() func to create the main vehicle and its module, to avoid incompatible with ray
        if engine_initialized():
            return
        self.engine = initialize_engine(self.config)
        # engine setup
        self.setup_engine()
        # other optional initialization
        self._after_lazy_init()

    def get_single_observation(self, _=None):
        o = TopDownMultiChannel(
            self.config["vehicle_config"],
            self.config["on_screen"],
            self.config["rgb_clip"],
            frame_stack=3,
            post_stack=10,
            frame_skip=1,
            resolution=(200, 200),
            max_distance=50
        )
        #o = TopDownMultiChannel(vehicle_config, self, False)
        return o
    
    def wrap_angle(self, angle_in_rad):
        #angle_in_rad = angle_in_degree / 180.0 * np.pi
        while (angle_in_rad > np.pi):
            angle_in_rad -= 2 * np.pi
        while (angle_in_rad <= -np.pi):
            angle_in_rad += 2 * np.pi
        return angle_in_rad


register(
    id='Control-v1',
    entry_point='core.envs.md_macro_env:MetaDriveControlEnv',
)
