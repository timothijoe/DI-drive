from tkinter.font import families
import metadrive
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from metadrive import TopDownMetaDrive
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import SACPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer
from core.envs import DriveEnvWrapper
#from core.policy.ad_policy.conv_qac i
# mport ConvQAC
from core.envs.md_traj_env import MetaDriveTrajEnv
from core.policy.hrl_policy.traj_vaiate_qac import ConvQAC 
from core.policy.hrl_policy.traj_sac import TrajSAC
from core.utils.simulator_utils.evaluator_utils import MetadriveEvaluator


ONE_SIDE_CLASS_VAE = False
LATENT_DIM = 3
TRAJ_CONTROL_MODE = 'acc' # 'acc', 'jerk'
SEQ_TRAJ_LEN = 20

if ONE_SIDE_CLASS_VAE:
    if LATENT_DIM == 3:
        VAE_LOAD_DIR = '/home/SENSETIME/zhoutong/hoffnung/xad/traj_model/var_len_zdim3_oneside_ckpt'
        #VAE_LOAD_DIR = 'traj_model/variate_len_dim3_v2_oneside_ckpt'
    else:
        VAE_LOAD_DIR = '/home/SENSETIME/zhoutong/hoffnung/xad/traj_model/var_len_zdim10_oneside_ckpt'
else:
    if LATENT_DIM == 3:
        VAE_LOAD_DIR = '/home/SENSETIME/zhoutong/hoffnung/xad/traj_model/multi_head_dim3_ckpt'
    else:
        VAE_LOAD_DIR = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/variate_len_dim10_noone_ckpt'    

metadrive_basic_config = dict(
    exp_name = 'metadrive_basic_sac',
    env=dict(
        metadrive=dict(use_render=True,
            show_seq_traj = True,
            traffic_density = 0.3,
            seq_traj_len = SEQ_TRAJ_LEN,
            traj_control_mode = TRAJ_CONTROL_MODE,
            #map='OSOS', 
            #map='XSXS',
            #map='SSSSSSS',
            #show_interface=False,
            use_lateral=True,
            use_speed_reward = True,
            use_heading_reward = True,
            use_jerk_reward = False,
            use_steer_rate_reward = True,
            use_theta_diff_reward = True,
            show_interface=False,
            avg_speed=6.5,
            driving_reward = 0.2, # 0.1
            speed_reward = 0.1, 
            heading_reward = 0.10, # 0.20
            jerk_importance = 0.8,
            sr_importance = 0.8,
            run_out_of_time_penalty = 10.0,
            extra_heading_penalty = True,
            print_debug_info = True,
            # const_episode_max_step = True,
            # episode_max_step = 250,
        ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=20,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=LATENT_DIM,
            vae_latent_dim = LATENT_DIM,
            encoder_hidden_size_list=[128, 128, 64],
            vae_seq_len = SEQ_TRAJ_LEN,
            vae_traj_control_mode = TRAJ_CONTROL_MODE,
            one_side_class_vae = ONE_SIDE_CLASS_VAE,
            vae_load_dir= VAE_LOAD_DIR, #'/home/SENSETIME/zhoutong/hoffnung/xad/ckpt_files/jerk_ckpt',
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(
            n_sample=50,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=10,
                )
            ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
            ),
        ), 
    )
)

main_config = EasyDict(metadrive_basic_config)


def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveTrajEnv(config=env_cfg), wrapper_cfg)


def main(cfg):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        SACPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
    )

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    evaluator_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    model = ConvQAC(**cfg.policy.model)
    policy = TrajSAC(cfg.policy, model=model)


    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    #collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    import torch
    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/may26/var_len_20k.pth.tar'
    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/may28/v4_var_20k.pth.tar'
    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/may28/vvv8_iter90k.pth.tar'
    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/june04_ondime10.pth.tar'
    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/june09_v2_1.pth.tar'
    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/june09_v2_2.pth.tar'
    # dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/jun09_v2_1.pth.tar'
    # dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/z5_june07.pth.tar'
    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june11/june09_11_v2_1.pth.tar'
    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june11/june13_v4_1.pth.tar'
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june11/june12_v3_1.pth.tar'
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june11/june12_v4_2.pth.tar'
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/june09_v3_1.pth.tar'
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/june09_v1_1.pth.tar'
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/jun06_iter20k.pth.tar'
    # dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/before_z1_oneside_dim10.pth.tar'
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june04/z4_noone_dim10.pth.tar'

    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june11/june17_v_1.pth.tar'
    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june11/june17_v2_1.pth.tar'
    dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/june11/june19_v2_1.pth.tar'
    
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/march23/b1_exp3/iteration_60000.pth.tar'
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/march26/c1_len15_exp3/c1_iteration_40000.pth.tar'
    #policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/stancy/ckpt_k8s/march12/exp1_jerk/iteration_70000.pth.tar', map_location = 'cpu'))
    
    policy._load_state_dict_collect(torch.load(dir, map_location = 'cpu'))
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
   # evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator.evall()
    # for iter in range(20):
    #     stop, reward = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
