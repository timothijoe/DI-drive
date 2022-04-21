import metadrive
import gym
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from core.envs import DriveEnvWrapper
from core.envs.md_imitation_env import MetaDriveImitationEnv



metadrive_basic_config = dict(
    exp_name='metadrive_basic_ppo',
    env=dict(
        metadrive=dict(
            use_render=True,
            use_sparse_reward = False,
            use_speed_reward = True,
            use_jerk_reward = True,
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
        action_space='continuous',
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=2,
            action_space='continuous',
            bound_type='tanh',
        ),
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(
            n_sample=300,
        ),
    ),
)

main_config = EasyDict(metadrive_basic_config)

def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveImitationEnv(env_cfg), wrapper_cfg)
# def wrapped_train_env(env_cfg):
#     env = gym.make("MetaDrive-1000envs-v0", config=env_cfg)
#     return DriveEnvWrapper(env)


# def wrapped_eval_env(env_cfg):
#     env = gym.make("MetaDrive-validation-v0", config=env_cfg)
#     return DriveEnvWrapper(env)


def main(cfg):
    cfg = compile_config(
        cfg, SyncSubprocessEnvManager, PPOPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator
    )

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    # collector_env = SyncSubprocessEnvManager(
    #     env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
    #     cfg=cfg.env.manager,
    # )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    policy = PPOPolicy(cfg.policy)
    import torch
    pathh = '/home/SENSETIME/zhoutong/drive_project/ckpt/april15/appo_iter_140k.pth.tar'
    #pathh = '/home/SENSETIME/zhoutong/hoffnung/xad/metadrive_basic_ppo/ckpt/iteration_10000.pth.tar'
    policy._load_state_dict_collect(torch.load(pathh, map_location = 'cpu'))

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    # collector = SampleSerialCollector(
    #     cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    # )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    learner.call_hook('before_run')

    for iter in range(5):
        stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, 1)
        if stop:
            break
        # Sampling data from environments
        # new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
        # learner.train(new_data, collector.envstep)
    learner.call_hook('after_run')

    #collector.close()
    evaluator.close()
    learner.close()


if __name__ == '__main__':
    main(main_config)