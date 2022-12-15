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
# from core.policy.hrl_policy.traj_qac import ConvQAC 
from core.policy.hrl_policy.traj_vaiate_qac import ConvQAC 
from core.policy.hrl_policy.traj_sac import TrajSAC
from core.utils.simulator_utils.evaluator_utils import MetadriveEvaluator



# ONE_SIDE_CLASS_VAE = True
TRAJ_CONTROL_MODE = 'acc'
SEQ_TRAJ_LEN = 20
VAE_LOAD_DIR = 'traj_model/oct28_len20_incontrol_ckpt'
VAE_LOAD_DIR='/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/result/Oct30-len20-v2-dim3/ckpt/30_decoder_ckpt'
VAE_LOAD_DIR='/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/result/Oct31-v3-len20-dim10/ckpt/99_decoder_ckpt'
VAE_LOAD_DIR='traj_model/oct_seq_len20_dim10_ckpt'
VAE_LOAD_DIR='/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/result/zNov-v2-len20-dim10/ckpt/99_decoder_ckpt'
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
            show_interface=False,
            show_interface_navi_mark=False,
            use_lateral=True,
            use_speed_reward = True,
            use_heading_reward = True,
            use_jerk_reward = True,
            avg_speed=6.0,
            print_debug_info = True,
            use_steer_rate_reward = True,
            heading_reward = 0.02,
            debug_info = True,
            # speed_reward = 0.015,
            # driving_reward=0.2,
            # sr_importance= 0.1,
        ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=1,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=10,
            vae_latent_dim=10,
            encoder_hidden_size_list=[128, 128, 64],
            vae_seq_len = SEQ_TRAJ_LEN,
            vae_traj_control_mode = TRAJ_CONTROL_MODE,
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
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/may10/exp3_straight_50k.pth.tar'
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/march23/b1_exp3/iteration_60000.pth.tar'
    dir = '/home/SENSETIME/zhoutong/luster/nov01/ckpt/len20/iteration_10000.pth.tar'
    #policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/stancy/ckpt_k8s/march12/exp1_jerk/iteration_70000.pth.tar', map_location = 'cpu'))


    policy._load_state_dict_collect(torch.load(dir, map_location = 'cpu'))
    #policy._load_state_dict_collect(torch.load(dir))
    #policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/stancy/ckpt_k8s/march12/jerk_full_reward/iteration_40000.pth.tar', map_location = 'cpu'))
    #policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/stancy/ckpt_k8s/march12/acc_full_reward/iteration_50000.pth.tar', map_location = 'cpu'))
    #policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/stancy/ckpt_k8s/march13/ours_no_lateral/iteration_40000.pth.tar', map_location = 'cpu'))
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    eval_mode = False
    # new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
    if eval_mode:
        evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    else:
        collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    for iter in range(5):
        if eval_mode:
            stop, reward = evaluator.eval()
        else:
            new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
    if eval_mode:
        evaluator.close()
    else:
        collector.close()


if __name__ == '__main__':
    main(main_config)
