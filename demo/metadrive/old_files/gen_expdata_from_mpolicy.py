import metadrive
import gym
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import DQNPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, AdvancedReplayBuffer
from ding.rl_utils import get_epsilon_greedy_fn
from core.envs import DriveEnvWrapper, MetaDriveMacroEnv
from core.envs.md_expert_env import MetaDriveExpertEnv
import os

num_expert_data_to_collect = 10
trex_path = 'test_trex_ad'
trex_reward_folder = trex_path + '/reward_model'
trex_expert_data_folder = trex_path + '/expert_data_folder'
trex_expert_ckpt_path = trex_path + '/expert_macro_policy/april10_iteration40k.pth.tar'

assert os.path.exists(trex_path)
assert os.path.exists(trex_expert_ckpt_path)

if not os.path.exists(trex_expert_data_folder):
    os.makedirs(trex_expert_data_folder)
# if not os.path.exists(trex_reward_folder):
#     os.makedirs(trex_reward_folder)
metadrive_macro_config = dict(
    exp_name='metadrive_macro_dqn',
    env=dict(
        metadrive=dict(
            use_render=True,
            show_interface=False,
            save_expert_data=False,
            expert_data_folder=trex_expert_data_folder,
            ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=2,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
        wrapper=dict(),
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=5,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            #epoch_per_collect=10,
            batch_size=64,
            learning_rate=1e-3,
            update_per_collect=100,
            hook=dict(
                load_ckpt_before_run='',
            ),
        ),
        collect=dict(
            n_sample=1000,
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
                replay_buffer_size=10000,
            ),
        ),
    ),
)

main_config = EasyDict(metadrive_macro_config)


def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveExpertEnv(env_cfg), wrapper_cfg)


def main(cfg):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )
    print(cfg.policy.collect.collector)

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    # collector_env = SyncSubprocessEnvManager(
    #     env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
    #     cfg=cfg.env.manager,
    # )
    evaluator_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    policy = DQNPolicy(cfg.policy)
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    #learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    #collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    
    #replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    import torch
    dir = trex_expert_ckpt_path
    policy._load_state_dict_collect(torch.load(dir))
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    for iter in range(num_expert_data_to_collect):
        stop, reward = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)