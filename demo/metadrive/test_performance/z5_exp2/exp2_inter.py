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
from core.policy.ad_policy.conv_qac import ConvQAC
from core.envs.md_hrl_env import MetaDriveHRLEnv
from core.envs.md_control_env import MetaDriveControlEnv
from core.utils.simulator_utils.evaluator_utils import MetadriveEvaluator
metadrive_basic_config = dict(
    exp_name = 'sac_only_jerk',
    env=dict(
        metadrive=dict(
            use_render=False,
            use_jerk_penalty = True,
            use_lateral_penalty = False,
            traffic_density = 0.3,
            #seq_traj_len = 1,
            seq_traj_len = 10,
            show_interface=False,
            #const_control = True,
            map ='XSXS',
            ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=50,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=4,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=20,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            update_per_collect=100,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(
            n_sample=5000,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
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
    return DriveEnvWrapper(MetaDriveControlEnv(config=env_cfg), wrapper_cfg)


def main(cfg):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        SACPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
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

    model = ConvQAC(**cfg.policy.model)
    policy = SACPolicy(cfg.policy, model=model)


    import os
    pwd = os.getcwd()
    file_path = pwd + 'iros_result/feb24/cluster61/z3_exp1_sac_round/iteration_5000.pth.tar'
    import torch
    # policy._load_state_dict_collect(torch.load(file_path, map_location = 'cpu'))

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    #learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    for iter in range(1):
        stop, reward = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)

