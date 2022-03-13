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
#from core.envs.md_control_env import MetaDriveControlEnv
#from core.envs.jerk_control_md_env import JerkControlMdEnv
from core.envs.md_traj_env import MetaDriveTrajEnv
from core.utils.simulator_utils.evaluator_utils import MetadriveEvaluator
#from core.policy.hrl_policy.traj_qac import ConvQAC 
from core.policy.hrl_policy.const_qac import ConstQAC 
from core.policy.hrl_policy.traj_sac import TrajSAC

TRAJ_CONTROL_MODE = 'acc' # 'acc', 'jerk'
SEQ_TRAJ_LEN = 10

metadrive_basic_config = dict(
    exp_name = 'az3_exp1_sac_inter',
    env=dict(
        metadrive=dict(
            use_render=True,
            seq_traj_len = SEQ_TRAJ_LEN,
            #use_jerk_penalty = True,
            #use_lateral_penalty = False,
            traffic_density = 0.35,
            traj_control_mode = TRAJ_CONTROL_MODE,
            use_lateral=False,
            use_speed_reward = True,
            use_heading_reward = False,
            use_jerk_reward = True,
            show_interface=False,
            ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=12,
        stop_value=99999,
        collector_env_num=20,
        evaluator_env_num=4,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            vae_traj_control_mode = TRAJ_CONTROL_MODE,
            vae_seq_len = SEQ_TRAJ_LEN,
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            auto_alpha=False,
        ),
        collect=dict(
            n_sample=10,
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
    return DriveEnvWrapper(MetaDriveTrajEnv(config=env_cfg), wrapper_cfg)


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
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    model = ConstQAC(**cfg.policy.model)
    policy = TrajSAC(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    #collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    #evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    #replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    import torch
    #policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/stancy/ckpt_k8s/march12/exp1_jerk/iteration_70000.pth.tar', map_location = 'cpu'))
    policy._load_state_dict_collect(torch.load('/home/SENSETIME/zhoutong/stancy/ckpt_k8s/march13/exp1_acc/iteration_60000.pth.tar', map_location = 'cpu'))
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    for iter in range(5):
        stop, reward = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
