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

class DDPG(object):
    def setup_placeholders(self):
        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        # placeholders
        # Prefixes and suffixes:
        # ob - observation
        # ac - action
        # _no - this tensor should have shape (batch size /n/, observation dim)
        # _na - this tensor should have shape (batch size /n/, action dim)
        # _n  - this tensor should have shape (batch size /n/)
        # placeholders前面都加一个前缀是好文明，可以方便在之后区分variable和placeholder
        self.sy_ob_no = tf.placeholder(tf.float32, shape=[None, ob_dim], name="ob")
        self.sy_ob_next = tf.placeholder(tf.float32, shape=[None, ob_dim], name="ob_next")
        self.terminal_next = tf.placeholder(tf.int32, shape=[None, 1], name="terminal_next")
        self.sy_rewards = tf.placeholder(tf.float32, shape=[None, 1], name="sy_rewards")
        self.sy_critic_targets = tf.placeholder(tf.float32, shape=[None, 1], name="sy_critic_targets")
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
        # actions的维度为整个actions选择的概率，而不只是输出一个被选择的action
        self.sy_actions = tf.placeholder(tf.float32, shape=[None, ac_dim], name='actions')

    def setup_network(self):
        # 指定Reuse即可reuse同一个scope下的网络参数
        self.actor_tf = build_actor(self.sy_ob_no, self.ac_dim, scope_name='actor')
        # target 输入下一次的ob
        self.target_actor_tf = build_actor(self.sy_ob_next, self.ac_dim, scope_name='target_actor')
        # 输入的是action的placeholder
        self.critic_tf = build_critic(self.sy_ob_no, self.sy_actions, scope_name='critic')
        # 输入的是模型的action概率分布，将这个输入到critic的第二层，也算是actor和critic共用一部分参数
        # critic_tf和critic_with_actor_tf使用同一个网络，只是输入的action不同
        self.critic_with_actor_tf = build_critic(self.sy_ob_no, self.actor_tf, scope_name='critic', reuse=True)
        # 计算next_q时用的是target_actor的输出作为actor部分的输入
        next_q = build_critic(self.sy_ob_next, self.target_actor_tf, scope_name='target_critic')
        self.target_q = self.sy_rewards + (1 - self.terminal_next) * self.gamma * next_q

        # setup loss
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        self.actor_update_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss)
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_tf, self.sy_critic_targets))
        self.critic_update_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

        # setup var updates
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
        critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')
        actor_init_updates, actor_soft_updates = get_target_updates(actor_vars, target_actor_vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(critic_vars, target_critic_vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def __init__(self,
                 exp_name='',
                 env_name='CartPole-v0',
                 gamma=1.0,

                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 logdir=None,
                 normalize_returns=True,
                 # network arguments
                 n_layers=1,
                 size=32,
                 gae_lambda=-1.0,
                 tau=0.001 #parameter update rate
                ):
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.normalize_returns = normalize_returns
        self.n_layers = n_layers
        self.size = size
        self.gae_lambda = gae_lambda
        self.tau = tau

        # Configure output directory for logging
        logz.configure_output_dir(logdir)
        # Log experimental parameters
        # args = inspect.getfullargspec(train_DDPG)[0]
        # locals_ = locals()
        # params = {k: locals_[k] if k in locals_ else None for k in args}
        # logz.save_params(params)

        # Make the gym environment
        self.env = gym.make(env_name)
        # Is this env continuous, or discrete?
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.setup_placeholders()
        self.setup_network()

    def train(self,
              seed=0,
              n_iter=100,
              animate=True,
              min_timesteps_per_batch=1000,
              batch_epochs=1,
              reward_to_go=True,
              max_path_length=None,
              ):
        start = time.time()
        # Set random seeds
        tf.set_random_seed(seed)
        np.random.seed(seed)
        # Maximum length for episodes
        max_path_length = max_path_length or self.env.spec.max_episode_steps
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

        sess = tf.Session(config=tf_config)
        sess.__enter__()  # equivalent to `with sess:`
        tf.global_variables_initializer().run()  # pylint: disable=E1101
        sess.run(self.target_init_updates)

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
                ob = self.env.reset()
                obs, acs, rewards, ob_nexts, dones = [], [], [], [], []
                animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
                steps = 0
                while True:
                    if animate_this_episode:
                        self.env.render()
                        time.sleep(0.05)
                    obs.append(ob)
                    ac, q = sess.run([self.actor_tf, self.critic_with_actor_tf], feed_dict={self.sy_ob_no: ob})
                    acs.append(ac)
                    ob_next, rew, done, _ = self.env.step(ac)
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