'''
created by zhenmaoli
2018-3-24
the action is defined as 10 news once
the state is defined as (index news,user information)

follow DQN
'''

import math

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.client import timeline

from ddqn import attention_tf
from models import model_nfm_batch as nfm
from utility.hype_parameters import HypeParameters as hypes


class ActorPickNewsQ():
    '''
    pick a number of news form candidates
    '''

    def __init__(self, tau, gamma,
                 batch_size,
                 num_pre_vars,  # trainable variables offset
                 optimizer,
                 max_userTags_num,
                 max_candidates_num,
                 learning_rate=0.0001,
                 cliped_norm=500,
                 num_history=50,
                 scope="Actor"):
        with tf.variable_scope(scope):
            self.tau = tau
            self.gamma = gamma
            self.batch_size = batch_size
            self.optimizer = optimizer
            self.max_candidates_num = max_candidates_num
            self.max_userTags_num = max_userTags_num
            self.num_history = 50

            self.action_recommended = tf.placeholder(dtype=tf.float32, shape=[batch_size, hypes.max_candidates_num,
                                                                              hypes.num_news_action],
                                                     name='action_in_record')

            # candidates_info
            self.candidates_tags = tf.placeholder(tf.int32, [batch_size, max_candidates_num, hypes.max_newsTag_num],
                                                  'candidates_input')
            self.candidates_mask = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_candidates_num],
                                                  name='candidates_mask')  # define mask
            # user_info
            self.user_tags = tf.placeholder(tf.int32, [batch_size, 1, max_userTags_num], "user_input")
            self.user_tags_w = tf.placeholder(tf.float32, [batch_size, 1, max_userTags_num], "user_input_w")
            self.user_info = (self.user_tags, self.user_tags_w, None)

            self.cliped_norm = cliped_norm * math.sqrt(hypes.num_news_action)

            # history_info
            self.history_newsTags = tf.placeholder(tf.int32, shape=[batch_size, num_history, hypes.max_newsTag_num],
                                                   name='history_newsTags')
            self.history_clicks = tf.placeholder(tf.int32, shape=[batch_size, num_history, 1],
                                                 name='history_clicks')
            self.history_positions = tf.placeholder(tf.int32, shape=[batch_size, num_history, 1],
                                                    name='history_positions')
            self.history = (self.history_newsTags, self.history_clicks, self.history_positions)

            self.grads = None

            # define mask
            mask_ones = []
            for i in range(max_candidates_num):
                mask_list_row = [0.0] * max_candidates_num
                mask_list_row[i] = 1.
                mask_ones.append(mask_list_row)
            self.indices_ones_mask = tf.constant(mask_ones, dtype=tf.float32)

            # Actor Network
            with tf.variable_scope('actor_Q_net') as actor_scope:
                self.nfm_model = nfm.NeuralFM(hypes.max_tags_num, hypes.tags_emb_size)

                self.candi_user_nfm = self.nfm_model.build_nfm_candidates_user_outerHistory_batch(self.candidates_tags,
                                                                                                  self.user_info,
                                                                                                  self.history,
                                                                                                  batch_size)
                self.candi_user_nfm = tf.check_numerics(self.candi_user_nfm, message="self.candi_user_nfm")

                self.nfm_model.news_embeddings = tf.check_numerics(self.nfm_model.news_embeddings,
                                                                   message="self.nfm_model.news_embeddings")
                tmp_embeding = tf.reshape(self.nfm_model.news_embeddings,
                                          shape=[batch_size, hypes.max_candidates_num, 16 * 64])
                self.news_attention = attention_tf.Attention(tmp_embeding, tmp_embeding,
                                                             tmp_embeding, nb_head=8,
                                                             size_per_head=64)
                self.news_attention = tf.check_numerics(self.news_attention, message="self.news_attention")
                self.model = tf.concat([self.candi_user_nfm, self.news_attention], axis=2)
                self.model = tf.check_numerics(self.model, message="self.model")
                init_w = tf.random_normal_initializer(0., 0.3)
                init_b = tf.constant_initializer(0.1)
                self.q_value = tf.layers.dense(self.model, 10, kernel_initializer=init_w,
                                               bias_initializer=init_b, name='a')
                self.q_value = tf.check_numerics(self.q_value, message="self.q_value")
                self.action_record_tmp, self.step_reward_tmp = self.pick_news_drop_batchALL(self.q_value,
                                                                                            self.action_recommended)
                self.value_estimate = tf.reduce_sum(self.q_value_net(self.step_reward_tmp), axis=1)
                self.network_params = tf.trainable_variables()[num_pre_vars:]
                actor_scope.reuse_variables()

            # Target Network
            with tf.variable_scope('target_Q_net') as target_scope:
                self.target_nfm_model = nfm.NeuralFM(hypes.max_tags_num, hypes.tags_emb_size)
                self.target_candi_user_nfm = self.target_nfm_model. \
                    build_nfm_candidates_user_outerHistory_batch(self.candidates_tags,
                                                                 self.user_info,
                                                                 self.history,
                                                                 batch_size)
                target_tmp_embeding = tf.reshape(self.target_nfm_model.news_embeddings,
                                                 shape=[batch_size, hypes.max_candidates_num, 16 * 64])
                self.target_news_attention = attention_tf.Attention(target_tmp_embeding, target_tmp_embeding,
                                                                    target_tmp_embeding, nb_head=8,
                                                                    size_per_head=64)
                print(self.target_candi_user_nfm)
                print(self.target_news_attention)
                init_w_target = tf.random_normal_initializer(0., 0.3)
                init_b_target = tf.constant_initializer(0.1)
                self.target_model = tf.concat([self.target_candi_user_nfm, self.target_news_attention], axis=2)
                self.q_target = tf.layers.dense(self.target_model, 10,
                                                kernel_initializer=init_w_target,
                                                bias_initializer=init_b_target, name='a')
                self.q_target = self.q_value_net(self.q_target)

            self.target_network_params = tf.trainable_variables()[num_pre_vars + len(self.network_params):]

            # Op for periodically updating target network with online network
            self.update_target_network_params = \
                [self.target_network_params[i].
                     assign(tf.multiply(self.network_params[i], self.tau) +
                            tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

            # Network target (y_i)
            self.y = tf.placeholder(tf.float32, shape=[batch_size], name='y')
            list_summary = []

            news_embeddings_summary = tf.summary.histogram("news_embeddings", self.nfm_model.news_embeddings)
            list_summary.append(news_embeddings_summary)
            target_news_embeddings_summary = tf.summary.histogram("target_news_embeddings",
                                                                  self.target_nfm_model.news_embeddings)
            list_summary.append(target_news_embeddings_summary)
            news_attention_summary = tf.summary.histogram("news_attention", self.news_attention)
            list_summary.append(news_attention_summary)
            target_news_attention_summary = tf.summary.histogram("target_news_attention", self.target_news_attention)
            list_summary.append(target_news_attention_summary)

            # Define loss and optimization Op
            self.loss = tf.losses.mean_squared_error(self.y, self.value_estimate)
            with tf.name_scope("weights_decay"):
                self.weight_decay = tf.add_n(
                    [hypes.weight_decay_rate * tf.nn.l2_loss(var) for var in self.network_params])

            with tf.variable_scope('adam', reuse=tf.AUTO_REUSE):
                self.train_op, self.grads = self.train_choice(total_loss=self.loss + self.weight_decay,
                                                              optimizer=self.optimizer,
                                                              learning_rate=learning_rate,
                                                              log_histograms=True,
                                                              list_summary=list_summary)

            self.param = tf.variable_scope

            self.num_trainable_vars = len(self.network_params) + \
                                      len(self.target_network_params)
            # tf.summmary

            # sv=tf.summary.scalar('value_estimate',tf.reduce_sum(self.value_estimate)/batch_size)
            # list_summary.append(sv)
            sl = tf.summary.scalar('value_loss', self.loss)
            list_summary.append(sl)

            s_weight_d = tf.summary.scalar('weight_decay', self.weight_decay)
            list_summary.append(s_weight_d)

            # weights about history
            hist_click_embedding = self.network_params[1]
            hist_click_embedding = tf.check_numerics(hist_click_embedding, message="hist_click_embedding")
            for i in range(3):
                sc = tf.summary.histogram('hist_click_' + str(i), hist_click_embedding[i])
                list_summary.append(sc)
            hist_position_embedding = self.network_params[2]
            for i in range(5):
                sp = tf.summary.histogram('hist_position_' + str(i), hist_position_embedding[i])
                list_summary.append(sp)

            self.merge = tf.summary.merge(list_summary)

    def train_choice(self, total_loss, optimizer, learning_rate, list_summary, log_histograms=True):

        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
        # Compute gradients.
        # with tf.control_dependencies([total_loss]):
        grads = opt.compute_gradients(total_loss)
        grads = [(g, v) for g, v in grads if g is not None]
        # clip
        with tf.name_scope("gradients_clips"):
            for i, (g, v) in enumerate(grads):
                if g is not None:
                    g = tf.check_numerics(g, message="grads[{%d}]" % i)
                    grads[i] = (tf.check_numerics(tf.clip_by_norm(g, 5), message="grads[%d]" % i), v)  # clip gradients
                else:
                    print("ERROR NULL grads")

                # Add histograms for gradients.
        if log_histograms:
            for grad, var in grads:
                if grad is not None:
                    if 'embedding' in var.name:
                        sg = tf.summary.histogram(var.name + '/gradients', grad)
                        sv = tf.summary.histogram(name=var.name, values=var)
                        list_summary.append(sg)
                        list_summary.append(sv)
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads)

        return apply_gradient_op, grads

    def train_choice_gradient(self, total_loss, optimizer, learning_rate, list_summary, log_histograms=True):

        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
        # Compute gradients.
        # with tf.control_dependencies([total_loss]):
        grads = opt.compute_gradients(total_loss)
        self.grads = grads

        res_grads = []

        # if log_histograms:
        #     for grad, var in grads:
        #         if grad is not None:
        # sg = tf.summary.histogram(var.op.name + '/gradients', grad)
        # sv = tf.summary.histogram(name=var.op.name, values=var)
        # list_summary.append(sg)
        # list_summary.append(sv)
        return opt, res_grads

    def pick_news_drop_batchALL(self, model, action):
        q_result = model * action
        q_result = tf.reduce_sum(q_result, axis=1)
        # q_result = tf.squeeze(q_result, axis=1)
        return action, q_result

    def pick_news_drop_batch(self, candi_user_nfm, candidates_tags, nfm_model, feed_action=False):
        '''
        :param candidates_tags: candidates news tags idx
        :param candi_user_nfm: nfm of candidate news and user information
        :return: a number of news idx
        '''
        res_positions = []
        res_rewards = []

        indices_mask_current = tf.constant(1.0, dtype=tf.float32, shape=[self.batch_size, self.max_candidates_num])
        batch_range = tf.reshape(tf.range(0, self.batch_size, dtype=tf.int32), [self.batch_size, 1])

        candi_user_nfm_trans = self.transNFMVector(candi_user_nfm)

        history_newsTags = tf.zeros(shape=[self.batch_size, hypes.num_news_action - 1, hypes.max_newsTag_num],
                                    dtype=tf.int32)
        history_newsTags_current = history_newsTags

        # define history position weights
        history_position_weights = tf.get_variable('inner_history_position_weights',
                                                   shape=[1, hypes.num_news_action - 1, 1, hypes.tags_emb_size],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        def step(candi_user_nfm_trans, candidates_tags, history_newsTags_current, indices_mask_current, i):

            candi_hist_nfm = self.build_candi_hist_nfm(candidates_tags, history_newsTags_current, nfm_model,
                                                       history_position_weights)
            news_probs_tmp = self.build_attention_all(candi_user_nfm_trans, candi_hist_nfm)

            # set picked position of news_probs to 0
            news_probs = tf.multiply(news_probs_tmp, indices_mask_current * self.candidates_mask, name='news_probs')
            tmp_position = tf.argmax(input=news_probs, axis=1, output_type=tf.int32, name="max_position")

            if feed_action == True:
                # tmp_position=self.action_recommended[:,i]
                tmp_position = tf.gather(self.action_recommended, i, axis=1, name='position_feed')
                # tmp_position = tf.convert_to_tensor(tmp_position,name='position_feed')

            picked_position_batch = tmp_position

            tmp_indices = tf.reshape(tmp_position, [-1, 1])
            indices_2d = tf.concat([batch_range, tmp_indices], 1)
            picked_prob_batch = tf.gather_nd(news_probs, indices_2d, name='picked_prob_batch')
            # picked_prob_batch=tf.squeeze(picked_prob_batch)

            # update indices_mask

            mask_ones_tmp = tf.gather(self.indices_ones_mask, tmp_position, axis=0)
            indices_mask_next = tf.subtract(indices_mask_current, mask_ones_tmp, name='indices_mask_next')

            history_newsTags_next = history_newsTags_current
            if i < hypes.num_news_action - 1:
                picked = tf.gather_nd(candidates_tags, indices_2d, name='picked_candidates_tags')
                picked_e = tf.expand_dims(picked, axis=1)
                # picked = tf.reshape(picked, [-1, hypes.max_newsTag_num])
                history_newsTags_next = tf.concat([history_newsTags_current[:, 0:i],
                                                   picked_e,
                                                   history_newsTags_current[:, i + 1:]], 1,
                                                  name='history_newsTags_next')

            return indices_mask_next, history_newsTags_next, picked_position_batch, picked_prob_batch

        with tf.variable_scope('for_steps') as for_scope:
            for_scope.reuse_variables()
            for i in range(hypes.num_news_action):
                with tf.name_scope('step'):
                    # run n times, if the number is not known, then we just need while_loop
                    indices_mask_current, \
                    history_newsTags_current, \
                    picked_position, \
                    picked_prob = step(candi_user_nfm_trans=candi_user_nfm_trans,
                                       candidates_tags=candidates_tags,
                                       history_newsTags_current=history_newsTags_current,
                                       indices_mask_current=indices_mask_current,
                                       i=i)

                    res_positions.append(picked_position)
                    res_rewards.append(picked_prob)

        return res_positions, res_rewards

    def predict_action(self, states, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []

        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []

        # mask
        candidates_mask_batch = []
        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])
            # print('c', len(s.candidates_tags))
            # print('u', len(s.user_tags))
            # print('uw', len(s.user_tags_w))
            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch}
        return sess.run(self.action_choose, feed_in)

    def predict_target_value2(self, states, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []

        # mask
        candidates_mask_batch = []
        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch}
        return sess.run(self.q_target, feed_in)

    def predict_target_value(self, states, action, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []

        # mask
        candidates_mask_batch = []
        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.action_recommended: action,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch}
        return sess.run(self.target_value_estimate, feed_in)

    def predict_target_action(self, states, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch}
        return sess.run(self.target_action, feed_in)

    def predict_target(self, states, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch}

        return sess.run([self.target_action,
                         self.target_action_reward,
                         self.target_action_value], feed_in)

    def predict_target_time(self, states, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch}

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        target_action, \
        target_action_reward, \
        tarrget_action_value = sess.run([self.target_action,
                                         self.target_action_reward,
                                         self.target_action_value], feed_in,
                                        options=run_options, run_metadata=run_metadata)
        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

        return target_action, target_action_reward, tarrget_action_value

    def train(self, states, action, y, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)
        sess = sess or tf.get_default_session()

        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.action_recommended: action,
                   self.y: y,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch}

        return sess.run([self.train_op,
                         self.loss,
                         self.value_estimate,
                         self.merge,
                         self.y,
                         self.candi_user_nfm,
                         self.grads,
                         self.q_value], feed_in)

    def train_time(self, states, action, y, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

        sess = sess or tf.get_default_session()

        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.action_recommended: action,
                   self.y: y,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch}

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        train_op, loss, value_estimate, step_reward, merge = sess.run([self.train_op,
                                                                       self.loss,
                                                                       self.value_estimate,
                                                                       self.step_reward,
                                                                       self.merge], feed_in,
                                                                      options=run_options, run_metadata=run_metadata)
        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('train_timeline.json', 'w') as f:
            f.write(ctf)

        return train_op, loss, value_estimate, step_reward, merge

    def update_target_network(self, sess=None):
        sess = sess or tf.get_default_session()
        sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def transNFMVector(self, candi_user_nfm):
        with tf.variable_scope('fc_for_candidate'):
            weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            candi_user_nfm = tf.expand_dims(candi_user_nfm, 1)  # [b,1,200,128]
            candidate_inter_1 = layers.conv2d(candi_user_nfm, 256,
                                              kernel_size=[1, 1],
                                              activation_fn=tf.nn.relu,
                                              weights_initializer=weights_initializer,
                                              weights_regularizer=weights_regularizer)

            candidate_inter_2 = layers.conv2d(candidate_inter_1, 128,
                                              kernel_size=[1, 1],
                                              activation_fn=tf.nn.relu,
                                              weights_initializer=weights_initializer,
                                              weights_regularizer=weights_regularizer)

        return tf.squeeze(candidate_inter_2, axis=1)  # [b,200,128]

    def build_candi_hist_nfm(self, candi_tags, hist_tags, nfm_model, history_position_weights):
        candi_hist_nfm = nfm_model.build_nfm_candidates_innerHistory_batch(candi_tags,
                                                                           hist_tags,
                                                                           history_position_weights,
                                                                           self.batch_size)
        candi_hist_nfm = tf.expand_dims(candi_hist_nfm, 1)
        with tf.variable_scope('fc_for_history', reuse=tf.AUTO_REUSE):
            weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            inter_1 = layers.conv2d(candi_hist_nfm, 256,
                                    kernel_size=[1, 1],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=weights_initializer,
                                    weights_regularizer=weights_regularizer)

            inter_2 = layers.conv2d(inter_1, 128,
                                    kernel_size=[1, 1],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=weights_initializer,
                                    weights_regularizer=weights_regularizer)
        return tf.squeeze(inter_2, axis=1)  # [b,200,128]

    def build_attention_all(self, candi_user_current, newsTags_history):

        with tf.variable_scope(name_or_scope='attention_all', reuse=tf.AUTO_REUSE):
            in_overall = tf.concat([candi_user_current, newsTags_history], 2)  # [b,200,128+128]
            # in_overall = tf.reshape(in_overall, [self.batch_size, 1, -1, hypes.tags_emb_size * 2])
            in_overall = tf.expand_dims(in_overall, 2)

            weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            tmp = layers.conv2d(in_overall, 128,
                                kernel_size=[1, 1],
                                activation_fn=tf.nn.relu,
                                weights_initializer=weights_initializer,
                                weights_regularizer=weights_regularizer)
            tmp1 = layers.conv2d(tmp, 1,
                                 kernel_size=[1, 1],
                                 activation_fn=tf.identity,
                                 weights_initializer=weights_initializer,
                                 weights_regularizer=weights_regularizer)
            tmp1 = tf.reshape(tmp1, [self.batch_size, -1])
        return tmp1
        # return tf.nn.softmax(tmp1)

    def q_value_net(self, action_rewards):
        with tf.variable_scope('q_out'):
            # in_overall = tf.concat([position,action_rewards], 1)
            action_rewards_cliped = tf.clip_by_norm(action_rewards, self.cliped_norm, axes=1)
        return action_rewards_cliped

    # debug#######################################################################################
    def show_tensor(self, states, name_list, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []

        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []

        # mask
        candidates_mask_batch = []
        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

        sess = sess or tf.get_default_session()
        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch}

        sess = sess or tf.get_default_session()
        tensor_list = []
        graph = sess.graph
        for n in name_list:
            tensor = graph.get_tensor_by_name(n)
            tensor_list.append(tensor)
        return sess.run(tensor_list, feed_in)

    def train_debug(self, states, action, y, sess=None):
        candidates_tags_batch = []
        user_tags_batch = []
        user_tags_w_batch = []
        # history
        h_newsTags_batch = []
        h_clicks_batch = []
        h_positions_batch = []
        # mask
        candidates_mask_batch = []

        for s in states:
            candidates_tags_batch.append(s.candidates_tags)
            user_tags_batch.append([s.user_tags])
            user_tags_w_batch.append([s.user_tags_w])

            h_newsTags = []
            h_clicks = []
            h_positions = []
            for h in s.history:
                h_newsTags.append(h[0])
                h_clicks.append([h[1]])
                h_positions.append([h[2]])
            h_newsTags_batch.append(h_newsTags)
            h_clicks_batch.append(h_clicks)
            h_positions_batch.append(h_positions)

            candidates_mask_batch.append(s.candidates_mask)

        sess = sess or tf.get_default_session()

        feed_in = {self.candidates_tags: candidates_tags_batch,
                   self.user_tags: user_tags_batch,
                   self.user_tags_w: user_tags_w_batch,
                   self.action_recommended: action,
                   self.y: y,
                   self.history_newsTags: h_newsTags_batch,
                   self.history_clicks: h_clicks_batch,
                   self.history_positions: h_positions_batch,
                   self.candidates_mask: candidates_mask_batch}

        fetch_list = [self.grads[12][0], self.grads[13][0], self.grads[14][0], self.grads[15][0],
                      self.loss, self.value_estimate, self.step_reward]

        return sess.run(fetch_list, feed_in)
