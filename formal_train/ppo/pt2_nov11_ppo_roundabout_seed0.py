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
from core.utils.simulator_utils.evaluator_utils import MetadriveEvaluator
from ding.utils import set_pkg_seed

TRAJ_CONTROL_MODE = 'acc' # 'acc', 'jerk'
SEQ_TRAJ_LEN = 1
EXP_SEED = 0

metadrive_basic_config = dict(
    exp_name='ppo_roundabout_seed0',
    env=dict(
        metadrive=dict(
            show_seq_traj = False,
            traffic_density = 0.3,
            seq_traj_len = SEQ_TRAJ_LEN,
            traj_control_mode = TRAJ_CONTROL_MODE,
            avg_speed = 6.5,
            use_lateral=True,
            use_speed_reward = True,
            use_heading_reward = True,
            use_jerk_reward = True,
            heading_reward=0.15,
            speed_reward = 0.05,
            driving_reward = 0.2,
            ignore_first_steer = False,
            map='OSOS',
            # map='SOSO',

            crash_vehicle_penalty = 4.0,
            out_of_road_penalty = 5.0,
            ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=20, #20
        stop_value=99999,
        collector_env_num=20, #20
        evaluator_env_num=10, #4
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=2,
            action_space='continuous',
            encoder_hidden_size_list=[128, 128, 64],
            vae_seq_len = SEQ_TRAJ_LEN,
            # vae_traj_control_mode = TRAJ_CONTROL_MODE,
            steer_rate_constrain_value=0.5,
        ),
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=3e-4,
            entropy_weight = 0.01,
            value_weight=0.5,
            clip_ratio = 0.05,
            adv_norm=False,
            value_norm=True,
            grad_clip_value=10,
        ),
        collect=dict(
            n_sample=500,
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
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    collector_env.seed(EXP_SEED)
    evaluator_env.seed(EXP_SEED, dynamic_seed=False)
    set_pkg_seed(EXP_SEED, use_cuda=cfg.policy.cuda)

    model = ConvVAC(**cfg.policy.model)
    policy = TrajPPO(cfg.policy, model=model)

    # policy = PPOPolicy(cfg.policy)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    # evaluator = InteractionSerialEvaluator(
    #     cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    # )

    learner.call_hook('before_run')

    while True:
        if evaluator.should_eval(learner.train_iter):
            #stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            stop, rate = evaluator.evall(learner.save_checkpoint, learner.train_iter, collector.envstep, collector._total_episode_count, collector._total_duration)
            if stop:
                break
        # Sampling data from environments
        new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
        learner.train(new_data, collector.envstep)

    collector.close()
    evaluator.close()
    learner.close()


if __name__ == '__main__':
    main(main_config)