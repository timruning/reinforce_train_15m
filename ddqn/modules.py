import tensorflow as tf
# import keras
from hyperparams import Hyperparams as hp
import numpy as np

def forward_conv_net(inputs,reuse=False):
    with tf.name_scope("conv_model"):
        outputs = tf.layers.conv1d(inputs, filters=50, kernel_size=3, padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   bias_initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        outputs = tf.layers.max_pooling1d(outputs, pool_size=2, strides=2)
        outputs = tf.nn.relu(outputs)

        outputs = tf.layers.conv1d(outputs,filters=20,kernel_size=3,padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   bias_initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        outputs = tf.layers.max_pooling1d(outputs,pool_size=2,strides=2)
        outputs = tf.nn.relu(outputs)
    return outputs


def lstm_embedding(inputs, num_units, batch_size, num_layers=1):
    # with tf.variable_scope("lstm"):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True)
    if num_layers > 1 :
        cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layers, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    _,last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)
    return last_state[1]


def k_max_dropout(inputs, k, activation=None):
    # inputs   b *n   2D
    epsilon = 0.0001
    with tf.name_scope("k_max_dropout"):
        top_k = tf.nn.top_k(inputs,k=k)
        min_top_k_value = tf.reduce_min(top_k[0],1)
        ones_mask = inputs >= tf.expand_dims(min_top_k_value,-1)
        zeros_mask = inputs < tf.expand_dims(min_top_k_value,-1)
        mask = tf.cast(ones_mask,tf.float32) + tf.cast(zeros_mask, tf.float32) * epsilon
        if None == activation:
            outputs = tf.multiply(inputs, mask)
        elif activation == "sigmoid":
            outputs = tf.multiply(tf.nn.sigmoid(tf.layers.batch_normalization(inputs)), mask)
        else:
            outputs = tf.multiply(inputs, mask)

    return outputs,mask


def k_max_pooling(inputs,k,name):
    # inputs b * n   2D
    # extract top_k, returns two tensors [values, indices]a
    #with tf.name_scope("k_max_pooling"):
    top_k = tf.nn.top_k(inputs,k=k,sorted=True,name=name)
    return top_k


def dense(inputs,units,activation=tf.nn.relu):
    # with tf.name_scope("dense"):
    outputs = tf.layers.dense(inputs, units, activation)
    return outputs


def sparse_connection(inputs, kernel, address_matrix):
    local_weights = kernel
    inputs = tf.transpose(inputs, [1, 0])
    value_weights = tf.nn.embedding_lookup(params=inputs, ids=address_matrix)  # N * k * B

    value_weights = tf.transpose(value_weights, [2, 0, 1])  # B * N * k
    # batch_size = value_weights.get_shape().as_list()[0]
    local_weights = tf.tile(tf.expand_dims(local_weights,0),[hp.batch_size, 1, 1])
    outputs = tf.reduce_sum(tf.multiply(local_weights,value_weights), axis=-1)
    return outputs


def sparse_conv1d(inputs, kernel, address_matrix):
    # the shape of kernel is same to the shape[1] of inputs
    inputs = tf.transpose(inputs, [1, 0]) # N * B
    value_weights = tf.nn.embedding_lookup(params=inputs, ids=address_matrix)  # N * k * B

    value_weights = tf.transpose(value_weights, [2, 0, 1])  # B * N * k
    kernel = tf.tile(tf.expand_dims(kernel, -1), [1, hp.batch_size])  # N * B
    local_weights = tf.nn.embedding_lookup(params=kernel, ids=address_matrix)  # N * k * B
    local_weights = tf.transpose(local_weights, [2, 0, 1])  # B * N * k
    outputs = tf.reduce_sum(tf.multiply(local_weights,value_weights), axis=-1)
    return outputs



def batch_normal(inputs,is_training):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    axis = list(range(len(inputs_shape)-1))

    return None


def idf_embedding(inputs, num_units=1):
    with tf.variable_scope("matrix", reuse=True):
        lookup_table = tf.get_variable("idf_matrix")

    lookup_table = tf.concat((tf.zeros(shape=[1,num_units], dtype=tf.float32), lookup_table[1:, :]), 0)

    outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    return outputs


def embedding(inputs, num_units=50, zero_pad=True, scale=True):
    with tf.variable_scope("embed_layer", reuse=True):
        lookup_table = tf.get_variable("embedding")
    if zero_pad:
        lookup_table = tf.concat((tf.zeros(shape=[1,num_units], dtype=tf.float32), lookup_table[1:, :]), 0)

    outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    if scale:
        outputs = outputs * (num_units ** 0.5)
    return outputs


def embedding_channel_2d(inputs, num_units=50, zero_pad=True, scale=True):
    with tf.variable_scope("embed_layer", reuse=True):
        word_embeddings = tf.get_variable("embedding")
        ft_word_embeddings = tf.get_variable("ft_embedding")
    if zero_pad:
        word_embeddings = tf.concat((tf.zeros(shape=[1, num_units], dtype=tf.float32), word_embeddings[1:, :]), 0)
        ft_word_embeddings = tf.concat((tf.zeros(shape=[1, num_units], dtype=tf.float32), ft_word_embeddings[1:, :]), 0)

    concat_embedds = tf.concat([tf.expand_dims(word_embeddings, -1), tf.expand_dims(ft_word_embeddings, -1)], axis=-1)

    outputs = tf.nn.embedding_lookup(concat_embedds, inputs)
    if scale:
        outputs = outputs * (num_units ** 0.5)
    return outputs


def contrastive_loss(y_true, y_pred, margin=1):
    # with tf.name_scope("contrastive_loss"):
    y_true = tf.cast(y_true, tf.float32)
    m1 = tf.square(y_pred)
    m2 = tf.square(tf.maximum(margin - y_pred, 0))
    loss = tf.reduce_mean(y_true * m1 + (1-y_true) * m2)
    return loss


def euclidean_distance(x,y):
    # with tf.name_scope("distance"):
    distance = tf.sqrt(tf.reduce_sum(tf.pow(x-y, 2), 1, keep_dims=True))
    return distance


# def cos_distance_sparse(x: dict, y: dict):
#     x_indices = x.keys()
#     y_indices = y.keys()
#     and_indices = set(x_indices).union(set(y_indices)) ^ (x_indices ^ y_indices)
#     # print(and_indices)
#     if 0 == len(and_indices):
#         inner = 0.0
#     else:
#         inner = np.sum([x[i] * y[i] for i in and_indices])
#     x_mode = np.sqrt(np.sum([np.square(x_v) for x_i, x_v in x.items()]))
#     y_mode = np.sqrt(np.sum([np.square(y_v) for y_i, y_v in y.items()]))
#     return inner/(x_mode * y_mode)


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

            # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, C)
        outputs = tf.layers.dense(outputs, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)

    return outputs

def idf_multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="weighted_multihead_attention",
                        reuse=None,
                        idf_embbed=None):
    '''Applies multihead attention.


    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

            # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        attention_score_matrix = outputs
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, C)
        outputs = tf.layers.dense(outputs, num_units, activation=tf.nn.relu)  # (N, T_q, C)

        # multi-head weighted Residual connection
        attention_score_matrix = tf.nn.softmax(attention_score_matrix,dim=1)
        residual_weighted = tf.reduce_mean(attention_score_matrix,axis=-1)  # (h*N, T_q)
        residual_weighted = tf.tile(tf.expand_dims(residual_weighted, axis=-1), [1, 1, num_units//num_heads])  # (h*N, T_q, C/h)
        residual_weighted = tf.concat(tf.split(residual_weighted, num_heads, axis=0), axis=2) # (N, T_q, C)

        outputs = tf.multiply(residual_weighted, outputs)

        outputs = tf.multiply(tf.tile(idf_embbed, [1, 1, num_units]), outputs)
        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)

    return outputs


def weighted_multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="weighted_multihead_attention",
                        reuse=None):
    '''Applies multihead attention.


    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

            # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        attention_score_matrix = outputs
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, C)
        outputs = tf.layers.dense(outputs, num_units, activation=tf.nn.relu)  # (N, T_q, C)

        # multi-head weighted Residual connection
        attention_score_matrix = tf.nn.softmax(attention_score_matrix,dim=1)
        residual_weighted = tf.reduce_mean(attention_score_matrix,axis=-1)  # (h*N, T_q)
        residual_weighted = tf.tile(tf.expand_dims(residual_weighted, axis=-1), [1, 1, num_units//num_heads])  # (h*N, T_q, C/h)
        residual_weighted = tf.concat(tf.split(residual_weighted, num_heads, axis=0), axis=2) # (N, T_q, C)
        outputs += tf.multiply(residual_weighted, queries)

        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)

    return outputs


def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs



def feedforward_var(inputs,
                num_units=[2048, 512],
                scope="feedforward_var",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += tf.layers.dense(inputs, units=num_units[1])

        # Normalize
        outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
     '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

     Args:
       inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
       epsilon: Smoothing rate.

     For example,

     ```
     import tensorflow as tf
     inputs = tf.convert_to_tensor([[[0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]],

       [[1, 0, 0],
        [1, 0, 0],
        [0, 1, 0]]], tf.float32)

     outputs = label_smoothing(inputs)

     with tf.Session() as sess:
         print(sess.run([outputs]))

     >>
     [array([[[ 0.03333334,  0.03333334,  0.93333334],
         [ 0.03333334,  0.93333334,  0.03333334],
         [ 0.93333334,  0.03333334,  0.03333334]],

        [[ 0.93333334,  0.03333334,  0.03333334],
         [ 0.93333334,  0.03333334,  0.03333334],
         [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
     ```
     '''
     K = inputs.get_shape().as_list()[-1] # number of channels
     return ((1-epsilon) * inputs) + (epsilon / K)



def positional_encoding(inputs,
                         num_units,
                         zero_pad=True,
                         scale=True,
                         scope="positional_encoding",
                         reuse=None):
     '''Sinusoidal Positional_Encoding.

     Args:
       inputs: A 2d Tensor with shape of (N, T).
       num_units: Output dimensionality
       zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
       scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
       scope: Optional scope for `variable_scope`.
       reuse: Boolean, whether to reuse the weights of a previous layer
         by the same name.

     Returns:
         A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
     '''

     N, T = inputs.get_shape().as_list()
     if N is None:
        N = hp.batch_size
     with tf.variable_scope(scope, reuse=reuse):
         position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

         # First part of the PE function: sin and cos argument
         position_enc = np.array([
             [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
             for pos in range(T)])

         # Second part, apply the cosine to even columns and sin to odds.
         position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
         position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

         # Convert to a tensor
         lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

         if zero_pad:
             lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                       lookup_table[1:, :]), 0)
         outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

         if scale:
             outputs = outputs * num_units**0.5

         return outputs


def channel_shuffle(inputs, shuffle_matrix):
    '''shuffle channel
             Args:
               inputs: A 3d tensor with shape of [N, T, C].
               shuffle_matrix: A 2d tensor with shape of [heads,C]
        '''
    inputs_trans = tf.transpose(inputs, [2, 0, 1])  # [C, N, T]
    N, T, C = inputs.get_shape().as_list()

    inputs_shuffle = tf.nn.embedding_lookup(params=inputs_trans, ids=shuffle_matrix)  #[heads, C, N, T]
    inputs_shuffle_trans = tf.transpose(inputs_shuffle,[2,1,3,0])  #[N, C, T, heads]
    inputs_shuffle_trans_reshape = tf.reshape(inputs_shuffle_trans,shape=[N,C,-1])  #[N, C, T * heads]
    return tf.transpose(inputs_shuffle_trans_reshape,[0, 2, 1])  #[N, T * heads, C]


def multi_head_token_shuffle(inputs, shuffle_matrix, horizontal=True):
    '''shuffle token
             Args:
               inputs: A 3d tensor with shape of [N, T, C].
               shuffle_matrix: A 2d tensor with shape of [heads,N]
        '''
    inputs_trans = tf.transpose(inputs, [1, 0, 2])  # [T, N, C]
    N, T, C = inputs.get_shape().as_list()
    inputs_shuffle = tf.nn.embedding_lookup(params=inputs_trans, ids=shuffle_matrix)  #[heads, T, N, C]
    if horizontal:
        inputs_shuffle_trans = tf.transpose(inputs_shuffle,[2,3,1,0])  #[N, C, T, heads]
        inputs_shuffle_trans_reshape = tf.reshape(inputs_shuffle_trans,shape=[N,C,-1])  #[N, C, T * heads]
        return tf.transpose(inputs_shuffle_trans_reshape,[0, 2, 1])  #[N, T * heads, C]
    else:
        inputs_shuffle_trans = tf.transpose(inputs_shuffle, [2, 1, 3, 0])  # [N, T, C, heads]
        inputs_shuffle_trans_reshape = tf.reshape(inputs_shuffle_trans, shape=[N, T, -1])  # [N, T, C * heads]
        return inputs_shuffle_trans_reshape



# def channel_shuffle(inputs):
#     '''shuffle channel
#          Args:
#            inputs: A 3d tensor with shape of [N, T, C].
#     '''
#     inputs_trans = tf.transpose(inputs, [2, 0, 1])
#     inputs_shuffle = tf.random_shuffle(inputs_trans)
#     return tf.transpose(inputs_shuffle, [1, 2, 0])

def bubble_sort(inputs):
    '''sort the 1st dim
         Args:
           inputs: A 3d tensor
    '''
    temp_list = []
    count = inputs.get_shape().as_list()[0]

    for i in range(0, count):
        temp_list.append(tf.maximum(inputs[i], inputs[i]))
    for i in range(0, count):
        for j in range(i + 1, count):
            max = tf.maximum(temp_list[i], temp_list[j])
            min = tf.minimum(temp_list[i], temp_list[j])
            temp_list[i] = min
            temp_list[j] = max
    return temp_list

if __name__=="__main__":
    a ={1:3.0,4:4.0}
    b = {5:4.0, 4:3.0}
    # print(cos_distance_sparse(a,b))
    # a=[1.0,0.0,1.0,0.0]
    # b =[0.9,0.1,0.2,0.2]
    # print(a)
    # with tf.Session() as sess:
    #     b=sess.run(k_max_pooling(tf.convert_to_tensor(a),b))
    #     print(b)
