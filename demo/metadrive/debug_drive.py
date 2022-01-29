import metadrive
import gym
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy, DDPGPolicy
#from ding.policy.ad_sac import ADSAC
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from core.envs import DriveEnvWrapper, MetaDriveMacroEnv
from core.envs.md_hrl_env import MetaDriveHRLEnv
from core.policy.ad_policy.hrl_sac import HRLSAC


z_freeze_vae = True
z_traj_seq_len = 30
z_vae_h_dim = 64
z_vae_latent_dim = 100
z_train_decoder = False



metadrive_rush_config = dict(
    exp_name = 'metadrive_macro_ppo3',
    env=dict(
        metadrive=dict(
            use_render=True,
        ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=3,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
        wrapper=dict(),
    ),
    policy=dict(
        cuda=False,
        traj_seq_len = z_traj_seq_len,
        #freeze_vae = z_freeze_vae,
        vae_h_dim = z_vae_h_dim,
        vae_latent_dim = z_vae_latent_dim, 
        train_decoder = z_train_decoder,
        continuous=False,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=100,
            action_space='regression',
            actor_head_hidden_size = 64,
            traj_seq_len = z_traj_seq_len,
            train_decoder = z_train_decoder,
            #continuous=True,
            #encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(
            n_sample=100,
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
                replay_buffer_size=1000,),
        ), 
    ),
)

main_config = EasyDict(metadrive_rush_config)


def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveHRLEnv(env_cfg), wrapper_cfg) #MetaDriveMacroEnv


def main(cfg):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        HRLSAC,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator
    )
    print(cfg.policy.collect.collector)

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    evaluator_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    policy = HRLSAC(cfg.policy)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    #evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)

    learner.call_hook('before_run')
    zt = 0

    while True:
        zt += 1
        if(zt % 100 == 0):
            print(zt)
        #print(zt)
        # if evaluator.should_eval(learner.train_iter):
        #     stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, 1)
        #     if stop:
        #         break
        #     if zt >=100:
        #         break
        # Sampling data from environments
        new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
        learner.train(new_data, collector.envstep)
    learner.call_hook('after_run')

    collector.close()
    evaluator.close()
    learner.close()

if __name__ == '__main__':
    main(main_config)