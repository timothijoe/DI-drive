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
from core.utils.simulator_utils.md_utils.traffic_manager_utils import TrafficMode
from core.envs.md_control_env import MetaDriveControlEnv
metadrive_basic_config = dict(
    exp_name='metadrive_tdv_ppo',
    env=dict(
        metadrive=dict(
        use_render=True,
        show_seq_traj=True,
        use_jerk_penalty = True,
        use_lateral_penalty = True,
        traffic_density = 0.3,
        seq_traj_len = 1,
        #map='SOSO', 
        #traffic_mode=TrafficMode.Trigger,
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
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            epoch_per_collect=1,
            batch_size=64,
            learning_rate=3e-4,
            hook=dict(load_ckpt_before_run = '/home/SENSETIME/zhoutong/hoffnung/xad/z4_exp1_ppo_straight_stepcor_2/ckpt/iteration_20000.pth.tar')
        ),
        collect=dict(
            n_sample=100,
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
    # evaluator_env = SyncSubprocessEnvManager(
    #     env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
    #     cfg=cfg.env.manager,
    # )
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    # evaluator_env = SyncSubprocessEnvManager(
    #     env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
    #     cfg=cfg.env.manager,
    # )

    model = ConvVAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))


    # import torch
    # policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/hoffnung/xad/z4_exp1_ppo_straight_stepcor_2/ckpt/iteration_800000.pth.tar', map_location = 'cpu'))

    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    # evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    # for iter in range(5):
    #     stop, reward = evaluator.eval()
    # evaluator.close()

    while True:
        #print("evaluator.should_eval(learner.train_iter): ", evaluator.should_eval(learner.train_iter))
        new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
    learner.call_hook('after_run')



if __name__ == '__main__':
    main(main_config)
