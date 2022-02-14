import metadrive
import gym
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy, DDPGPolicy
#from ding.policy.ad_sac import ADSAC
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, AdvancedReplayBuffer
from core.envs import DriveEnvWrapper, MetaDriveMacroEnv
from core.envs.md_hrl_env import MetaDriveHRLEnv
from core.policy.ad_policy.traj_sac import TrajSAC
import os

z_freeze_decoder = False
z_use_wp_decoder = False
z_traj_seq_len = 30
z_vae_h_dim = 64
z_vae_latent_dim = 100
z_dt = 0.03


pwd = os.getcwd()
ckpt_path = 'ckpt_files/vae_decoder_ckpt'
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
    exp_name = 'traj_len30_nord',
    env=dict(
        metadrive=dict(
            use_render=False,
            show_seq_traj = False,
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
        collector_env_num=11,
        evaluator_env_num=4,
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
                replay_buffer_size=100000,
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
    return DriveEnvWrapper(MetaDriveHRLEnv(env_cfg), wrapper_cfg) #MetaDriveMacroEnv


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
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    policy = TrajSAC(cfg.policy)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    eps_cfg = cfg.policy.other.eps
    from ding.rl_utils import get_epsilon_greedy_fn
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    learner.call_hook('before_run')
    zt = 0

    while True:
        zt += 1
        print('total {} times'.format(zt))
        if evaluator.should_eval(learner.train_iter):
            stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Sampling data from environments
        eps = epsilon_greedy(collector.envstep)
        #new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter,policy_kwargs={'eps': eps})
        new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                break
            learner.train(train_data, collector.envstep)
    learner.call_hook('after_run')

    collector.close()
    evaluator.close()
    learner.close()

if __name__ == '__main__':
    main(main_config)