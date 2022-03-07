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
from core.policy.ad_policy.conv_vac import ConvVAC
from core.envs.md_hrl_env import MetaDriveHRLEnv
from ding.policy import PPOPolicy
from core.envs.md_control_env import MetaDriveControlEnv

# from core.envs.md_control_env import MetaDriveControlEnv
from core.utils.simulator_utils.evaluator_utils import MetadriveEvaluator

metadrive_basic_config = dict(
    exp_name='metadrive_tdv_ppo',
    env=dict(
        metadrive=dict(
        use_render=False,
        show_seq_traj=False,
        traffic_density = 0.3,
        use_jerk_penalty = True,
        use_lateral_penalty = True,
        seq_traj_len = 1,
        map ='OSOS',
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
        action_space='continuous',
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=2,
            action_space='continuous',
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(
            n_sample=1000,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
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
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
    )

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    # evaluator_env = SyncSubprocessEnvManager(
    #     env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
    #     cfg=cfg.env.manager,
    # )

    model = ConvVAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)


    import torch
    policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/hoffnung/xad/feb28_ckpt/z1_ppo/round/iteration_90000.pth.tar', map_location = 'cpu'))


    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    #learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    for iter in range(1):
        stop, reward = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
