import numpy as np
from pg_utils import *
import tensorflow as tf
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process

class PPO(object):
    def __init__(self,
                 env_name='CartPole-v0',
                 gamma=1.0,
                 max_path_length=None,
                 learning_rate=5e-3,
                 logdir=None,
                 normalize_advantages=True,
                 nn_baseline=False,
                 # network arguments
                 n_layers=1,
                 size=32,
                 gae_lambda=-1.0,
                 model_tag='vanilla',
                 #ppo parameter
                 clip_ratio=0.2,):
        #params
        self.nn_baseline = nn_baseline
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        self.n_layers = n_layers
        self.size = size
        self.gae_lambda = gae_lambda
        self.model_tag = model_tag
        self.clip_ratio = clip_ratio
        # Configure output directory for logging
        logz.configure_output_dir(logdir)
        self.log_dir = logdir
        # Log experimental parameters
        # args = inspect.getfullargspec(__init__)[0]
        # locals_ = locals()
        # params = {k: locals_[k] if k in locals_ else None for k in args}
        # logz.save_params(params)

        # Make the gym environment
        self.env = gym.make(env_name)
        # Is this env continuous, or discrete?
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Maximum length for episodes
        self.max_path_length = max_path_length or self.env.spec.max_episode_steps
        self.setup_placeholders()
        self.setup_tf_operations()
        self.setup_loss()
        if self.nn_baseline:
            self.setup_baseline()

    def setup_placeholders(self):
        # Observation and action sizes
        self.ob_dim = self.env.observation_space.shape[0]
        self.ac_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        self.sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            self.sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            self.sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        # Define a placeholder for advantages
        self.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

    def setup_tf_operations(self):
        if self.discrete:
            # YOUR_CODE_HERE
            scope_name = 'discrete'
            old_scope_name = 'discrete_old'
            self.sy_logits_na = build_mlp(self.sy_ob_no, self.ac_dim, scope_name, self.n_layers, self.size)
            # softmax生成prob被压缩在sparse_softmax_cross_entropy_with_logits中，提升效率
            # 因此sy_logits_na是没有归一化的，但不影响分布sample的生成
            self.sy_sampled_ac = tf.reshape(tf.multinomial(self.sy_logits_na, 1), [-1])  # Hint: Use the tf.multinomial op
            # 这里加负号为了兼容 continuous的情况，loss也加负号
            self.sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sy_ac_na, logits=self.sy_logits_na)

            self.old_logits_na = build_mlp(self.sy_ob_no, self.ac_dim, old_scope_name, self.n_layers, self.size)
            self.old_sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sy_ac_na, logits=self.old_logits_na)
        else:
            # YOUR_CODE_HERE
            scope_name = 'continuous'
            old_scope_name = 'continuous_old'
            self.sy_mean = build_mlp(self.sy_ob_no, self.ac_dim, scope_name, self.n_layers, self.size)
            # logstd should just be a trainable variable, not a network output.
            # ??? why
            self.sy_logstd = tf.get_variable('std', [self.ac_dim], dtype=tf.float32)
            self.sy_sampled_ac = tf.random_normal(shape=tf.shape(self.sy_mean), mean=self.sy_mean, stddev=self.sy_logstd)
            # Hint: Use the log probability under a multivariate gaussian.
            self.sy_logprob_n = tf.contrib.distributions.MultivariateNormalDiag(loc=self.sy_mean,
                                                         scale_diag=tf.exp(self.sy_logstd)).log_prob(self.sy_ac_na)

            self.old_sy_mean = build_mlp(self.sy_ob_no, self.ac_dim, old_scope_name, self.n_layers, self.size)
            self.old_sy_logprob_n = tf.contrib.distributions.MultivariateNormalDiag(loc=self.old_sy_mean,
                                                                               scale_diag=tf.exp(self.sy_logstd)).log_prob(self.sy_ac_na)

        self.old_network_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, old_scope_name)
        self.network_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_name)
        self.param_assign_op = [tf.assign(old_value, new_value) for (old_value, new_value) in
                           zip(self.old_network_param, self.network_param)]

    def setup_loss(self):
        if self.model_tag == 'ppo':
            # 和tensorforce不同 这里stop_gradient之后的梯度为0，导致lossDelta为0
            # old_log_prob = tf.stop_gradient(input=sy_logprob_n)
            prob_ratio = tf.exp(x=(self.sy_logprob_n - self.old_sy_logprob_n))
            # 这里无法指定axis=1 因为只有一维，剩下的一维就是[?] 即batch_size
            prob_ratio = tf.reduce_mean(input_tensor=prob_ratio)
            clipped_prob_ratio = tf.clip_by_value(
                t=prob_ratio,
                clip_value_min=(1.0 - self.clip_ratio),
                clip_value_max=(1.0 + self.clip_ratio)
            )
            self.loss = tf.reduce_mean(-tf.minimum(x=(prob_ratio * self.sy_adv_n), y=(clipped_prob_ratio * self.sy_adv_n)))
        else:  # vanilla pg
            self.loss = tf.reduce_mean(-self.sy_logprob_n * self.sy_adv_n)

        tf.summary.scalar('loss', self.loss)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def setup_baseline(self):
        self.baseline_prediction = tf.squeeze(build_mlp(
            self.sy_ob_no,
            1,
            "nn_baseline",
            n_layers=self.n_layers,
            size=self.size))
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        self.baseline_targets = tf.placeholder(shape=[None], name='baseline_targets', dtype=tf.float32)
        self.baseline_loss = tf.nn.l2_loss(self.baseline_prediction - self.baseline_targets)
        tf.summary.scalar('baseline_loss', self.baseline_loss)
        self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.baseline_loss)

    def train(self,
              n_iter=100,
              seed=0,
              animate=True,
              min_timesteps_per_batch=1000,
              batch_epochs=1,
              reward_to_go=True,):
        start = time.time()
        # Set random seeds
        tf.set_random_seed(seed)
        np.random.seed(seed)

        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        sess = tf.Session(config=tf_config)
        sess.__enter__()  # equivalent to `with sess:`
        tf.global_variables_initializer().run()  # pylint: disable=E1101
        total_timesteps = 0
        merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        for itr in range(n_iter):
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
                obs, acs, rewards = [], [], []
                animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
                steps = 0
                while True:
                    if animate_this_episode:
                        self.env.render()
                        time.sleep(0.05)
                    obs.append(ob)
                    ac = sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob[None]})
                    ac = ac[0]
                    acs.append(ac)
                    ob, rew, done, _ = self.env.step(ac)
                    rewards.append(rew)
                    steps += 1
                    if done or steps > self.max_path_length:
                        break
                path = {"observation": np.array(obs),
                        "reward": np.array(rewards),
                        "action": np.array(acs)}
                paths.append(path)
                timesteps_this_batch += pathlength(path)
                if timesteps_this_batch > min_timesteps_per_batch:
                    break
            total_timesteps += timesteps_this_batch

            # Build arrays for observation, action for the policy gradient update by concatenating
            # across paths
            ob_no = np.concatenate([path["observation"] for path in paths])
            ac_na = np.concatenate([path["action"] for path in paths])

            # YOUR_CODE_HERE
            q_n = []
            reward_n = []
            for path in paths:
                reward = path['reward']
                max_step = len(reward)
                reward_n.extend(reward)
                # 从当前t开始的value估算
                if reward_to_go:
                    q = [np.sum(np.power(self.gamma, np.arange(max_step - t)) * reward[t:]) for t in range(max_step)]
                else:  # 整个trajectory的q值估算
                    q = [np.sum(np.power(self.gamma, np.arange(max_step)) * reward) for t in range(max_step)]
                q_n.extend(q)

            epoch_step = 1
            for epoch in range(batch_epochs):
                # ====================================================================================#
                #                           ----------SECTION 5----------
                # Computing Baselines
                # ====================================================================================#
                # print('run %d epoch' % epoch)
                if self.nn_baseline:
                    # If nn_baseline is True, use your neural network to predict reward-to-go
                    # at each timestep for each trajectory, and save the result in a variable 'b_n'
                    # like 'ob_no', 'ac_na', and 'q_n'.
                    #
                    # Hint #bl1: rescale the output from the nn_baseline to match the statistics
                    # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
                    # #bl2 below.)
                    b_n = sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob_no})
                    # b_n_norm = b_n - np.mean(b_n, axis=0) / (np.std(b_n, axis=0) + 1e-7)
                    # 这里b_n要根据qn设置回来，因为b_n在下面optimize时是标准化过的
                    b_n = b_n * np.std(q_n, axis=0) + np.mean(q_n, axis=0)

                    if self.gae_lambda > 0:
                        adv_n = lambda_advantage(reward_n, b_n, len(reward_n), self.gae_lambda * self.gamma)
                    else:
                        adv_n = q_n - b_n
                else:
                    adv_n = q_n.copy()

                # ====================================================================================#
                #                           ----------SECTION 4----------
                # Advantage Normalization
                # ====================================================================================#

                if self.normalize_advantages:
                    # On the next line, implement a trick which is known empirically to reduce variance
                    # in policy gradient methods: normalize adv_n to have mean zero and std=1.
                    # YOUR_CODE_HERE
                    adv_mean = np.mean(adv_n, axis=0)
                    adv_std = np.std(adv_n, axis=0)
                    adv_n = (adv_n - adv_mean) / (adv_std + 1e-7)

                # ====================================================================================#
                #                           ----------SECTION 5----------
                # Optimizing Neural Network Baseline
                # ====================================================================================#
                if self.nn_baseline:
                    # ----------SECTION 5----------
                    # If a neural network baseline is used, set up the targets and the inputs for the
                    # baseline.
                    #
                    # Fit it to the current batch in order to use for the next iteration. Use the
                    # baseline_update_op you defined earlier.
                    #
                    # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
                    # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)
                    # 标准化的q_n作为baseline的优化目标
                    q_n_mean = np.mean(q_n, axis=0)
                    q_n_std = np.std(q_n, axis=0)
                    q_n = (q_n - q_n_mean) / (q_n_std + 1e-7)
                    sess.run(self.baseline_update_op, feed_dict={self.sy_ob_no: ob_no, self.baseline_targets: q_n})

                # ====================================================================================#
                #                           ----------SECTION 4----------
                # Performing the Policy Update
                # ====================================================================================#

                # Call the update operation necessary to perform the policy gradient update based on
                # the current batch of rollouts.
                #
                # For debug purposes, you may wish to save the value of the loss function before
                # and after an update, and then log them below.
                # 输出两次loss是为了下面的log
                feed_dict = {self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_adv_n: adv_n}
                sess.run(self.param_assign_op, feed_dict)
                #loss_1 = sess.run(self.loss, feed_dict)
                _, summary_val = sess.run([self.update_op, merged_summary], feed_dict)
                #loss_2 = sess.run(self.loss, feed_dict)
                global_step = itr*batch_epochs+epoch_step
                epoch_step = epoch_step+1
                self.summary_writer.add_summary(summary_val, global_step)
                #self.summary_writer.flush()
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

        self.summary_writer.flush()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', '-algo', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
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

    max_path_length = args.ep_len if args.ep_len > 0 else None

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
    ppo = PPO(
        env_name=args.env_name,
        gamma=args.discount,
        max_path_length=max_path_length,
        learning_rate=args.learning_rate,
        logdir=os.path.join(logdir, '%d' % seed),
        normalize_advantages=not (args.dont_normalize_advantages),
        nn_baseline=args.nn_baseline,
        n_layers=args.n_layers,
        size=args.size,
        gae_lambda=args.gae_lambda,
        model_tag=args.exp_name
    )
    ppo.train(n_iter=args.n_iter,
              seed=seed,
              min_timesteps_per_batch=args.batch_size,
              reward_to_go=args.reward_to_go,
              animate=args.render,
              batch_epochs=args.batch_epochs,)

if __name__ == "__main__":
    main()