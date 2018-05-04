import tensorflow as tf
import tensorflow.contrib as tc

def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None
):
    # ========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units.
    #
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    # ========================================================================================#

    with tf.variable_scope(scope):
        inputs = input_placeholder
        for i in range(n_layers):
            inputs = tf.layers.dense(inputs=inputs, units=size, activation=activation)
        output = tf.layers.dense(inputs=inputs, units=output_size, activation=output_activation)
        return output

def build_actor(state_input, output_size,
                scope_name='actor', n_layers=2, hidden_size=64, norm=True, reuse=False,
                activation=tf.nn.relu, output_activation=tf.tanh):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        x = state_input
        for i in range(n_layers):
            #不设activation则是线性层，因为中间要插layer_norm所以要单独设置
            x = tf.layers.dense(x, hidden_size)
            if norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = activation(x)
        # activation直接传入func tf.tanh等价于tf.nn.tanh
        x = tf.layers.dense(x, output_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), activation=output_activation)

def build_critic(state_input, action_input,
                scope_name='critic', n_layers=2, hidden_size=64, norm=True, reuse=False,
                activation=tf.nn.relu):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        x = state_input
        for i in range(n_layers):
            # 不设activation则是线性层，因为中间要插layer_norm所以要单独设置
            x = tf.layers.dense(x, hidden_size)
            if norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = activation(x)
            #第一层之后传入action作为输入
            if i == 1:
                x = tf.concat([x, action_input], axis=-1)
        # critic 最终直接输出为线性层
        x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

def get_target_updates(vars, target_vars, tau):
    #logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        #logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)

def pathlength(path):
    return len(path["reward"])


def lambda_advantage(reward, value, length, discount):
    """Generalized Advantage Estimation."""
    timestep = tf.range(reward.shape[1].value)
    mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
    next_value = tf.concat([value[:, 1:], tf.zeros_like(value[:, -1:])], 1)
    delta = reward + discount * next_value - value
    advantage = tf.reverse(tf.transpose(tf.scan(
        lambda agg, cur: cur + discount * agg,
        tf.transpose(tf.reverse(mask * delta, [1]), [1, 0]),
        tf.zeros_like(delta[:, -1]), 1, False), [1, 0]), [1])
    return tf.check_numerics(tf.stop_gradient(advantage), 'advantage')