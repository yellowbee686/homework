import numpy as np
import tensorflow as tf
import gym
from . import logz
#import logz
import time
import inspect
from .pg_utils import *

#todo add action_noise
#todo add every normalize denormalize
#todo mpi, optimizer minimize to grad and update, for mpi

def train_DDPG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             actor_lr=1e-4,
             critic_lr=1e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             normalize_returns=True,
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             gae_lambda=-1.0,
             batch_epochs=1,
             tau=0.001
             ):
    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getfullargspec(train_DDPG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # placeholders
    # Prefixes and suffixes:
    # ob - observation
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # placeholders前面都加一个前缀是好文明，可以方便在之后区分variable和placeholder
    sy_ob_no = tf.placeholder(tf.float32, shape=[None, ob_dim], name="ob")
    sy_ob_next = tf.placeholder(tf.float32, shape=[None, ob_dim], name="ob_next")
    terminal_next = tf.placeholder(tf.int32, shape=[None, 1], name="terminal_next")
    sy_rewards = tf.placeholder(tf.float32, shape=[None, 1], name="sy_rewards")
    sy_critic_targets = tf.placeholder(tf.float32, shape=[None, 1], name="sy_critic_targets")
    param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
    # actions的维度为整个actions选择的概率，而不只是输出一个被选择的action
    sy_actions = tf.placeholder(tf.float32, shape=[None, ac_dim], name='actions')

    # 指定Reuse即可reuse同一个scope下的网络参数
    actor_tf = build_actor(sy_ob_no, ac_dim, scope_name='actor')
    # target 输入下一次的ob
    target_actor_tf = build_actor(sy_ob_next, ac_dim, scope_name='target_actor')
    # 输入的是action的placeholder
    critic_tf = build_critic(sy_ob_no, sy_actions, scope_name='critic')
    # 输入的是模型选择的action
    critic_with_actor_tf = build_critic(sy_ob_no, actor_tf, scope_name='critic', reuse=True)
    # 下一个state按照模型选择的q值
    next_q = build_critic(sy_ob_next, target_actor_tf, scope_name='target_critic')
    target_q = sy_rewards + (1 - terminal_next) * gamma * next_q


    # setup loss
    actor_loss = -tf.reduce_mean(critic_with_actor_tf)
    actor_update_op = tf.train.AdamOptimizer(actor_lr).minimize(actor_loss)
    critic_loss = tf.reduce_mean(tf.square(critic_tf, sy_critic_targets))
    critic_update_op = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss)

    #setup var updates
    actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
    target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
    critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
    target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')
    actor_init_updates, actor_soft_updates = get_target_updates(actor_vars, target_actor_vars, tau)
    critic_init_updates, critic_soft_updates = get_target_updates(critic_vars, target_critic_vars, tau)
    target_init_updates = [actor_init_updates, critic_init_updates]
    target_soft_updates = [actor_soft_updates, critic_soft_updates]

    # ========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    # ========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    sess = tf.Session(config=tf_config)
    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101
    sess.run(target_init_updates)

    # ========================================================================================#
    # Training Loop
    # ========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)

        # Collect paths until we have enough timesteps
        # 每一轮结束或者超过max_path_length时会结束一次path
        # 每一轮path结束后填充到paths中，检查一次总的batch步数是否超过batch需求数，超过了则退出，开始训练
        # 因此每次训练的都是完整的数据

        # PG算法每次都使用当前分布sample action，不涉及exploration
        # TODO 改成observation和train分开两个进程，这样不用互相等待
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards, ob_nexts, dones = [], [], [], [], []
            animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                ac, q = sess.run([actor_tf, critic_with_actor_tf], feed_dict={sy_ob_no: ob})
                acs.append(ac)
                ob_next, rew, done, _ = env.step(ac)
                ob_nexts.append(ob_next)
                dones.append(done)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation": np.array(obs),
                    "reward": np.array(rewards),
                    "action": np.array(acs),
                    "ob_next": np.array(ob_nexts),
                    "done": np.array(dones)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])



        # todo train process
        # todo memory sample in paths
        for epoch in range(batch_epochs):
            # Log diagnostics
            returns = [path["reward"].sum() for path in paths]
            ep_lengths = [pathlength(path) for path in paths]
            #logz.log_tabular("LossDelta", loss_1 - loss_2)
            logz.log_tabular("Time", time.time() - start)
            logz.log_tabular("Iteration", itr)
            logz.log_tabular("AverageReturn", np.mean(returns))
            logz.log_tabular("StdReturn", np.std(returns))
            logz.log_tabular("MaxReturn", np.max(returns))
            logz.log_tabular("MinReturn", np.min(returns))
            logz.log_tabular("EpLenMean", np.mean(ep_lengths))
            logz.log_tabular("EpLenStd", np.std(ep_lengths))
            logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
            logz.log_tabular("TimestepsSoFar", total_timesteps)
            logz.dump_tabular()
            logz.pickle_tf_vars()