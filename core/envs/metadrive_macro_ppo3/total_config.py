exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 2,
            'retry_type': 'reset',
            'auto_reset': True,
            'step_timeout': None,
            'reset_timeout': None,
            'retry_waiting_time': 0.1,
            'cfg_type': 'BaseEnvManagerDict',
            'shared_memory': False,
            'context': 'spawn'
        },
        'metadrive': {
            'use_render': True,
            'seq_traj_len': 30,
            'physics_world_step_size': 0.03,
            'show_seq_traj': True
        },
        'n_evaluator_episode': 3,
        'stop_value': 99999,
        'collector_env_num': 1,
        'evaluator_env_num': 1,
        'wrapper': {}
    },
    'policy': {
        'model': {
            'twin_critic':
            True,
            'action_space':
            'reparameterization',
            'obs_shape': [5, 200, 200],
            'action_shape':
            100,
            'actor_head_hidden_size':
            64,
            'freeze_decoder':
            True,
            'vae_seq_len':
            30,
            'vae_latent_dim':
            100,
            'vae_h_dim':
            64,
            'vae_dt':
            0.03,
            'vae_load_dir':
            '/home/SENSETIME/zhoutong/hoffnung/xad/result/vae_decoder_ckpt'
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'multi_gpu': False,
            'update_per_collect': 1,
            'batch_size': 64,
            'learning_rate_q': 0.0003,
            'learning_rate_policy': 0.0003,
            'learning_rate_value': 0.0003,
            'learning_rate_alpha': 0.0003,
            'target_theta': 0.005,
            'discount_factor': 0.99,
            'alpha': 0.2,
            'auto_alpha': True,
            'log_space': True,
            'ignore_done': False,
            'init_w': 0.003,
            'epoch_per_collect': 10,
            'learning_rate': 0.0003
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict'
            },
            'collector_logit': False,
            'n_sample': 100,
            'unroll_len': 1
        },
        'eval': {
            'evaluator': {
                'eval_freq': 50,
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'stop_value': 99999,
                'n_episode': 3
            }
        },
        'other': {
            'replay_buffer': {
                'replay_buffer_size': 1000
            },
            'eps': {
                'type': 'exp',
                'start': 0.95,
                'end': 0.1,
                'decay': 10000
            }
        },
        'type': 'sac',
        'cuda': False,
        'on_policy': False,
        'priority': False,
        'priority_IS_weight': False,
        'random_collect_size': 10000,
        'multi_agent': False,
        'cfg_type': 'TrajSACDict',
        'freeze_decoder': True,
        'continuous': False
    },
    'exp_name': 'metadrive_macro_ppo3',
    'seed': 0
}
