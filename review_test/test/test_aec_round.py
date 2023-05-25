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
from core.utils.simulator_utils.md_utils.traffic_manager_utils import TrafficMode


# ONE_SIDE_CLASS_VAE = True
TRAJ_CONTROL_MODE = 'acc'
SEQ_TRAJ_LEN = 10
VAE_LOAD_DIR = 'traj_model/oct28_len20_incontrol_ckpt'
VAE_LOAD_DIR='/home/SENSETIME/zhoutong/hoffnung/xad/traj_model/oct_seq_len10_dim3_nov1_ckpt'
VAE_LOAD_DIR='/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/result/zNov-v7-len10-dim3-kllager/ckpt/99_decoder_ckpt'
VAE_LOAD_DIR = '/mnt/lustre/zhoutong/august/xad/traj_model/oct_seq_len10_dim3_nov1_ckpt'
VAE_LOAD_DIR = 'traj_model/oct_seq_len10_dim3_nov1_ckpt'
VAE_LOAD_DIR = 'traj_model/oct_seq_len10_dim3_withtan_nov1_ckpt'
VAE_LOAD_DIR="traj_model/nov02_len10_dim3_v1_ckpt"
# VAE_LOAD_DIR='/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/result/zNov-v5-len10-dim3-notanh-kllager/ckpt/95_decoder_ckpt'
# VAE_LOAD_DIR='traj_model/oct30_v2_decoder_ckpt'
from ding.utils import set_pkg_seed
EXPP_SEED = 1

metadrive_basic_config = dict(
    exp_name = 'metadrive_basic_sac',
    seed = EXPP_SEED,
    env=dict(
        metadrive=dict(use_render=True,
            show_seq_traj = True,
            traffic_density = 0.30, #0.20
            # need_inverse_traffic=True, #True

            seq_traj_len = SEQ_TRAJ_LEN,
            traj_control_mode = TRAJ_CONTROL_MODE,
            
            # map='SOSO', 
            map='OSOS',
            # map='SXSX',
            # map = 'XSXS',
            # map='SXSX',
            # enable_u_turn = True,
            # traffic_mode=TrafficMode.Trigger,
            avg_speed = 6.0,


            #show_interface=False,
            use_lateral=True,
            use_speed_reward = True,
            use_heading_reward = True,
            use_jerk_reward = True,
            heading_reward=0.15,
            speed_reward = 0.05,
            driving_reward = 0.2,
            jerk_bias = 10,

            crash_vehicle_penalty = 4.0,
            out_of_road_penalty = 5.0,
            debug_info=True,
            ignore_first_steer = False,
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
            action_shape=3,
            vae_latent_dim=3,
            encoder_hidden_size_list=[128, 128, 64],
            vae_seq_len = SEQ_TRAJ_LEN,
            vae_traj_control_mode = TRAJ_CONTROL_MODE,
            vae_load_dir= VAE_LOAD_DIR, #'/home/SENSETIME/zhoutong/hoffnung/xad/ckpt_files/jerk_ckpt',
            steer_rate_constrain_value=0.5,
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



    # collector_env.seed(cfg.seed)
    # evaluator_env.seed(cfg.seed, dynamic_seed=False)
    # set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)


    collector_env.seed(EXPP_SEED)
    evaluator_env.seed(EXPP_SEED, dynamic_seed=False)
    set_pkg_seed(EXPP_SEED, use_cuda=cfg.policy.cuda)

    model = ConvQAC(**cfg.policy.model)
    policy = TrajSAC(cfg.policy, model=model)


    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    #collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    import torch
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/may10/exp3_straight_50k.pth.tar'
    #dir = '/home/SENSETIME/zhoutong/drive_project/ckpt/march23/b1_exp3/iteration_60000.pth.tar'
    dir = '/home/SENSETIME/zhoutong/luster/nov02/nov02_len10_dim3_v1_ckpt/iteration_6000.pth.tar'
    dir = '/home/SENSETIME/zhoutong/luster/nov03/straight/iteration_10000.pth.tar'
    #dir='/home/SENSETIME/zhoutong/Downloads/iteration_50000.pth.tar'
    dir='/home/SENSETIME/zhoutong/luster/nov4/round/iteration_25000.pth.tar'
    dir='/home/SENSETIME/zhoutong/luster/nov4/inter/iteration_14008.pth.tar'
    dir = '/home/SENSETIME/zhoutong/luster/nov4/inter/try_12000_with_uturn.pth.tar'
    dir = '/home/SENSETIME/zhoutong/luster/nov7/ckpt/inter_uturn/iteration_18000.pth.tar'
    dir = '/home/zhoutong/hoffung/ztaecrl_ckpt/iteration_70000_round.pth.tar'

    policy._load_state_dict_collect(torch.load(dir, map_location = 'cpu'))
    #policy._load_state_dict_collect(torch.load(dir))
    #policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/stancy/ckpt_k8s/march12/jerk_full_reward/iteration_40000.pth.tar', map_location = 'cpu'))
    #policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/stancy/ckpt_k8s/march12/acc_full_reward/iteration_50000.pth.tar', map_location = 'cpu'))
    #policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/stancy/ckpt_k8s/march13/ours_no_lateral/iteration_40000.pth.tar', map_location = 'cpu'))
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    #new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    #collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    for iter in range(15):
        stop, reward = evaluator.eval()
        # new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
    evaluator.close()
    # collector.close()


if __name__ == '__main__':
    main(main_config)
