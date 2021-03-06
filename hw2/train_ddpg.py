import numpy as np
from pg_utils import *
import tensorflow as tf
import gym
import logz
import time
import inspect
from memory import Memory
import os

#todo add action_noise
#todo add every normalize denormalize
#todo mpi, optimizer minimize to grad and update, for mpi

class DDPG(object):
    def setup_placeholders(self):
        # placeholders
        # Prefixes and suffixes:
        # ob - observation
        # ac - action
        # _no - this tensor should have shape (batch size /n/, observation dim)
        # _na - this tensor should have shape (batch size /n/, action dim)
        # _n  - this tensor should have shape (batch size /n/)
        # placeholders前面都加一个前缀是好文明，可以方便在之后区分variable和placeholder
        self.sy_ob_no = tf.placeholder(tf.float32, shape=[None, self.ob_dim], name="ob")
        self.sy_ob_next = tf.placeholder(tf.float32, shape=[None, self.ob_dim], name="ob_next")
        self.terminal_next = tf.placeholder(tf.float32, shape=[None, 1], name="terminal_next")
        self.sy_rewards = tf.placeholder(tf.float32, shape=[None, 1], name="sy_rewards")
        self.sy_critic_targets = tf.placeholder(tf.float32, shape=[None, 1], name="sy_critic_targets")
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
        # actions的维度为整个actions选择的概率，而不只是输出一个被选择的action
        # tensorforce是按以下实现的，因此应该也是输入的概率
        # x_actions = tf.reshape(tf.cast(x_actions, dtype=tf.float32), (-1, 1))
        # 现在的问题是cartpole返回的shape是一个() 空tuple 但实际上应该是两个action 感觉应该是个bug
        self.sy_actions = tf.placeholder(tf.float32, shape=[None, self.ac_dim], name='actions')

    def setup_network(self):
        # 指定Reuse即可reuse同一个scope下的网络参数 self.actor返回一个tensor，表示选择每个action的概率
        self.actor_tf = build_actor(self.sy_ob_no, self.ac_dim, scope_name='actor')
        # 默认axis为0 会返回[0,0]，即沿着第0维归一化，由于只有一个数，因此固定返回0,0
        # tf.argmax返回一个[1] 的tensor 使用tf.squeeze规约为int 否则env.step会检查不通过
        self.actor_choose_action = tf.squeeze(tf.argmax(self.actor_tf, axis=1))
        # target 输入下一次的ob
        self.target_actor_tf = build_actor(self.sy_ob_next, self.ac_dim, scope_name='target_actor')

        # 输入的是action的placeholder 这里可以选择输入action的选择概率，即actor_network的原始输出
        # 也可以选择输入argmax之后的action index
        self.critic_tf = build_critic(self.sy_ob_no, self.sy_actions, scope_name='critic')
        # 输入的是模型的action概率分布，将这个输入到critic的第二层，也算是actor和critic共用一部分参数
        # critic_tf和critic_with_actor_tf使用同一个网络，只是输入的action不同
        self.critic_with_actor_tf = build_critic(self.sy_ob_no, self.actor_tf, scope_name='critic', reuse=True)
        # 计算next_q时用的是target_actor的输出作为actor部分的输入
        next_q = build_critic(self.sy_ob_next, self.target_actor_tf, scope_name='target_critic')
        # terminal_next如果是tf.int32的placeholder，则会报 *
        self.target_q = self.sy_rewards + (1 - self.terminal_next) * self.gamma * next_q

        # setup var updates
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
        critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')
        actor_init_updates, actor_soft_updates = get_target_updates(actor_vars, target_actor_vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(critic_vars, target_critic_vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

        # setup loss
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        # 构造AdamOptimizer.minimize之后，会让actor_vars中的参数膨胀两倍，因此需要先设置updates，再设置loss
        self.actor_update_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss)
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_tf-self.sy_critic_targets))
        self.critic_update_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)


    def __init__(self,
                 env=None,
                 discrete=True,
                 ob_shape=(),
                 ac_dim=0,
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
        self.critic_lr = critic_lr
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
        self.env = env
        # Is this env continuous, or discrete?
        self.discrete = discrete
        self.ac_dim = ac_dim
        self.ob_dim = ob_shape[0]
        #observation_shape in cartpole is (2,) 一个tuple
        self.memory = Memory(limit=int(1e6), action_shape=ac_dim, observation_shape=ob_shape)
        self.setup_placeholders()
        self.setup_network()

    def sample_action(self,obs,compute_Q=True):
        feed_dict = {self.sy_ob_no:[obs]}
        # baseline的代码中这里直接输出action的选择概率，而且传入env.step时乘以env.high 应该是用于连续action的做法
        # 而我们求argmax则是用于离散action的做法
        # build_critic(self.sy_ob_no, self.actor_tf, scope_name='critic', reuse=True) critic传入的是actor的输出
        if compute_Q:
            action, action_prob, q = self.sess.run([self.actor_choose_action, self.actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action, action_prob = self.sess.run([self.actor_choose_action, self.actor_tf], feed_dict=feed_dict)
            q = None

        # 去除多余的维度，并限制在-1到1之间
        action_prob = action_prob.flatten()
        action_prob = np.clip(action_prob, -1., 1.)
        return action, action_prob, q

    def soft_sync_target_actor(self):
        self.sess.run(self.target_soft_updates)

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        self.memory.append(obs0, action, reward, obs1, terminal1)

    # 相当于baseline.ddpg.train 执行一次更新，
    def update_loss(self):
        batch = self.memory.sample(batch_size=self.batch_size)

        target_Q = self.sess.run(self.target_q, feed_dict={
            self.sy_ob_next: batch['obs1'],
            self.sy_rewards: batch['rewards'],
            self.terminal_next: batch['terminals1'].astype('float32'),
        })
        ops = [self.actor_loss, self.critic_loss, self.actor_update_op, self.critic_update_op]
        actor_loss, critic_loss, _, _ = self.sess.run(ops, feed_dict={
            self.sy_ob_no: batch['obs0'],
            self.sy_actions: batch['actions'],
            self.sy_critic_targets: target_Q,
        })

        return critic_loss, actor_loss

    # 完整的训练流程
    def train(self,
              seed=0,
              n_iter=100,
              animate=False,
              min_timesteps_per_batch=1000,
              batch_epochs=1,
              batch_size = 32,
              max_path_length=None,
              ):
        self.batch_size = batch_size
        start = time.time()
        # Set random seeds
        tf.set_random_seed(seed)
        np.random.seed(seed)
        # Maximum length for episodes
        max_path_length = max_path_length
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

        sess = tf.Session(config=tf_config)
        self.sess = sess
        sess.__enter__()  # equivalent to `with sess:`
        tf.global_variables_initializer().run()  # pylint: disable=E1101
        sess.run(self.target_init_updates)
        # todo: use finalize to make sure no new node in graph
        #sess.graph.finalize() #make it readonly, speed up
        # ========================================================================================#
        # Training Loop
        # ========================================================================================#
        #max_action = self.env.action_space.high
        total_timesteps = 0

        for itr in range(n_iter):
            #print('start train itr=%d max_step=%d batch=%d'%(itr, max_path_length, min_timesteps_per_batch))
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
                #obs, acs, ac_probs, rewards, ob_nexts, dones = [], [], [], [], [], []
                obs, acs, rewards, ob_nexts, dones = [], [], [], [], []
                animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
                steps = 0
                while True:
                    if animate_this_episode:
                        self.env.render()
                        time.sleep(0.05)
                    obs.append(ob)
                    # eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)
                    # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    # baseline将action限制在-1,1 再scale 可以看下这样是否有必要
                    if self.discrete:
                        ac, ac_prob, q = self.sample_action(ob, False)
                        acs.append(ac)
                        ob_next, rew, done, _ = self.env.step(ac)
                    else:
                        _, ac_prob, q = self.sample_action(ob, False)
                        #ac_prob = tf.Print(ac_prob, [ac_prob, ac_prob.shape], 'sample action')
                        acs.append(ac_prob)
                        ob_next, rew, done, _ = self.env.step(ac_prob)
                    #ac_probs.append(ac_prob)

                    ob_nexts.append(ob_next)
                    dones.append(done)
                    rewards.append(rew)
                    self.store_transition(ob, ac_prob, rew, ob_next, done)
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
            epoch_actor_losses = []
            epoch_critic_losses = []
            for epoch in range(batch_epochs):
                cl, al = self.update_loss()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                self.soft_sync_target_actor()
                # Log diagnostics
                returns = [path["reward"].sum() for path in paths]
                ep_lengths = [pathlength(path) for path in paths]
                #print('log iter %d'%itr)
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--actor_learning_rate', '-alr', type=float, default=5e-5)
    parser.add_argument('--critic_learning_rate', '-clr', type=float, default=5e-4)
    parser.add_argument('--critic_update_tau', '-tau', type=float, default=0.001)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--seed', '-seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--gae_lambda', '-gae', type=float, default=-1.0)
    parser.add_argument('--batch_epochs', '-be', type=int, default=1)
    args = parser.parse_args()

    if not (os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    env = gym.make(args.env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    max_path_length = args.ep_len if args.ep_len > 0 else env.spec.max_episode_steps
    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # for e in range(args.n_experiments):
    #     seed = args.seed + 10*e
    #     print('Running experiment with seed %d'%seed)
    #     def train_func():
    #         train_PG(
    #             exp_name=args.exp_name,
    #             env_name=args.env_name,
    #             n_iter=args.n_iter,
    #             gamma=args.discount,
    #             min_timesteps_per_batch=args.batch_size,
    #             max_path_length=max_path_length,
    #             learning_rate=args.learning_rate,
    #             reward_to_go=args.reward_to_go,
    #             animate=args.render,
    #             logdir=os.path.join(logdir,'%d'%seed),
    #             normalize_advantages=not(args.dont_normalize_advantages),
    #             nn_baseline=args.nn_baseline,
    #             seed=seed,
    #             n_layers=args.n_layers,
    #             size=args.size
    #             )
    #     # Awkward hacky process runs, because Tensorflow does not like
    #     # repeatedly calling train_PG in the same thread.
    #     p = Process(target=train_func, args=tuple())
    #     p.start()
    #     p.join()

    seed = args.seed
    print('Running experiment with seed %d' % seed)
    ddpg = DDPG(
        env=env,
        discrete=discrete,
        ac_dim=ac_dim,
        ob_shape=ob_shape,
        gamma = args.discount,
        actor_lr=args.actor_learning_rate,
        critic_lr=args.critic_learning_rate,
        logdir=os.path.join(logdir, '%d' % seed),
        normalize_returns=not (args.dont_normalize_advantages),
        # network arguments
        n_layers=args.n_layers,
        size=args.size,
        gae_lambda=args.gae_lambda,
        tau=args.critic_update_tau,
    )
    ddpg.train(
        n_iter=args.n_iter,
        seed=seed,
        min_timesteps_per_batch=args.batch_size,
        animate=args.render,
        batch_epochs=args.batch_epochs,
        batch_size=args.batch_size,
        max_path_length=max_path_length
    )

if __name__ == "__main__":
    main()