import metadrive
import gym
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from core.envs import DriveEnvWrapper, MetaDriveMacroEnv
import os 
from core.envs.md_hrl_env import MetaDriveHRLEnv
from core.policy.ad_policy.traj_sac import TrajSAC
z_freeze_decoder = True
z_use_wp_decoder = False
z_traj_seq_len = 10
z_vae_h_dim = 64
z_vae_latent_dim = 2
z_dt = 0.1


pwd = os.getcwd()
ckpt_path = 'ckpt_files/a79_decoder_ckpt'
ckpt_path = os.path.join(pwd, ckpt_path)
z_episode_max_step = 500
print(ckpt_path)
if z_traj_seq_len == 1:
    z_episode_max_step = z_episode_max_step * 30
if z_use_wp_decoder:
    z_vae_latent_dim = 2 * z_traj_seq_len
    z_freeze_decoder = False
if z_freeze_decoder:
    ckpt_path = ckpt_path 
else:
    ckpt_path = None

print(ckpt_path)
metadrive_rush_config = dict(
    exp_name = 'traj_len30_freeze',
    env=dict(
        metadrive=dict(
            use_render=True,
            show_seq_traj = True,
            seq_traj_len = z_traj_seq_len,
            physics_world_step_size = z_dt,
            episode_max_step = z_episode_max_step
            
        ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=12,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
        wrapper=dict(),
    ),
    policy=dict(
        cuda=True,         
        freeze_decoder = z_freeze_decoder,
        continuous=False,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=z_vae_latent_dim,
            action_space='reparameterization',
            actor_head_hidden_size = 64,
            freeze_decoder = z_freeze_decoder,
            vae_seq_len = z_traj_seq_len,
            vae_latent_dim = z_vae_latent_dim,
            vae_h_dim = z_vae_h_dim,
            vae_dt = z_dt,
            vae_load_dir = ckpt_path,
            use_wp_decoder = z_use_wp_decoder,
        ),
        learn=dict(
            update_per_collect=100,
            #epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            auto_alpha = False,
            alpha = 0.5,
            learner = dict(
                hook = dict(
                    save_ckpt_after_iter = 100,
                )
            )
        ),
        collect=dict(
            n_sample=5000,
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(
                replay_buffer_size=50000,
                deepcopy = False,
                max_use = float("inf"),
                max_staleness = float("inf"),
                alpha = 0.6,
                beta = 0.4, 
                anneal_step = 100000,
                thruput_controller = {'push_sample_rate_limit': {'max': float("inf"), 'min': 0}, 'window_seconds': 30, 'sample_min_limit_ratio': 1},
                monitor={'sampled_data_attr': {'average_range': 5, 'print_freq': 200}, 'periodic_thruput': {'seconds': 60}},
                enable_track_used_data=False,
                ),
        ), 
    ),
)

main_config = EasyDict(metadrive_rush_config)


def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveHRLEnv(env_cfg), wrapper_cfg)

def main(cfg):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        TrajSAC,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator
    )
    print(cfg.policy.collect.collector)

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    policy = TrajSAC(cfg.policy)
    import torch
    policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/hoffnung/xad/iteration_ckpt/iteration_99900.pth.tar', map_location = 'cpu'))


    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    #learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    for iter in range(5):
        stop, reward = evaluator.eval()
    evaluator.close()

if __name__ == '__main__':
    main(main_config)