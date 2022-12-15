import metadrive
import gym
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy
from core.policy.hrl_policy.traj_ppo import TrajPPO
# from core.policy.hrl_policy.conv_vac import ConvVAC
from core.policy.hrl_policy.traj_vac import ConvVAC
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from core.envs import DriveEnvWrapper
#from core.envs.md_imitation_env import MetaDriveImitationEnv
from core.envs.md_traj_env import MetaDriveTrajEnv

TRAJ_CONTROL_MODE = 'acc' # 'acc', 'jerk'
SEQ_TRAJ_LEN = 1

metadrive_basic_config = dict(
    exp_name='ppo_dense25_with_jerk',
    env=dict(
        metadrive=dict(
            use_render = True,
            seq_traj_len = SEQ_TRAJ_LEN,
            traffic_density=0.30,
            use_sparse_reward = False,
            use_speed_reward = True,
            use_jerk_reward = True,
            ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=11,
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

            encoder_hidden_size_list=[128, 128, 64],
            vae_seq_len = SEQ_TRAJ_LEN,
            # vae_traj_control_mode = TRAJ_CONTROL_MODE,
            steer_rate_constrain_value=0.5,
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(
            n_sample=64,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
            ),
        ),
    ),
)

main_config = EasyDict(metadrive_basic_config)

def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveTrajEnv(env_cfg), wrapper_cfg)
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
    # collector_env = BaseEnvManager(
    #     env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
    #     cfg=cfg.env.manager,
    # )
    evaluator_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )


    model = ConvVAC(**cfg.policy.model)
    policy = TrajPPO(cfg.policy, model=model)
    import torch
    dir = '/home/SENSETIME/zhoutong/luster/before nov13/ppo_ckpt_iter79k.pth.tar'

    policy._load_state_dict_collect(torch.load(dir, map_location = 'cpu'))

    # policy = PPOPolicy(cfg.policy)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    # collector = SampleSerialCollector(
    #     cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    # )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    learner.call_hook('before_run')

    while True:
        stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, 1)
        # if evaluator.should_eval(learner.train_iter):
        #     stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, 1)
        #     if stop:
        #         break
        # Sampling data from environments
        #new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
        #learner.train(new_data, collector.envstep)
    learner.call_hook('after_run')

    collector.close()
    evaluator.close()
    learner.close()


if __name__ == '__main__':
    main(main_config)