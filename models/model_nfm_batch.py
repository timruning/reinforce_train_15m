'''
created by zhenmaoli
2018.3.12
reference:Xiangnan He, Tat-Seng Chua. Neural Factorization
Machines for Sparse Predictive Analytics. In Proc. of SIGIR 2017.
nfm for processing the news and user information
'''
import tensorflow as tf
import tensorflow.contrib.layers as layers


class NeuralFM():
    def __init__(self, max_feature_num, embedding_size, num_click=3, num_history=7, num_userTagType=5):
        self.max_feature_num = max_feature_num
        self.embedding_size = embedding_size
        self.num_click = num_click
        self.num_history = num_history
        self.num_userTagType = num_userTagType

        # with tf.device('/device:GPU:0'):
        with tf.variable_scope("tags_lookup"):
            lookup_table = tf.get_variable('tag_lookup_table',
                                           shape=[max_feature_num, embedding_size],
                                           dtype=tf.float32,
                                           # initializer=tf.zeros_initializer())
                                           initializer=layers.xavier_initializer())

            # zero pad ,if word id is zero, then the embedding is zero vector
            lookup_table = tf.concat((tf.zeros(shape=[1, embedding_size], dtype=tf.float32), lookup_table[1:, :]), 0)

            self.lookup_table = lookup_table

        self.num_trainable_vars = len(tf.trainable_variables())
        print(self.num_trainable_vars)

    def position_embedding(self,
                           inputs,
                           vocab_size,
                           num_units,
                           scope="position_embedding",
                           reuse=tf.AUTO_REUSE):

        with tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable('position_lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           initializer=tf.contrib.layers.xavier_initializer())

            lookup_table_c = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table_c, inputs)

        return outputs

    def build_nfm_batch(self, embedding_1, embedding_2):
        # sum_embedding_1
        sum_embeddings_1 = tf.reduce_sum(embedding_1, 2)  # [batch_size,1,128]
        # sum_embedding_2
        sum_embeddings_2 = tf.reduce_sum(embedding_2, 2)  # [batch_size,none,128]

        summed_features_emb = sum_embeddings_1 + sum_embeddings_2  # [batch_size,none,128]

        summed_features_emb_square = tf.square(summed_features_emb)  # [batch_size,none,128]
        # _________ square_sum part _____________
        # square_sum_embedding_1
        embeddings_1_square = tf.square(embedding_1)  # [batch_size,1,m,128]
        embeddings_1_square_sum = tf.reduce_sum(embeddings_1_square, 2)  # [batch_size,1,128]
        # square_sum_embedding_2
        embeddings_2_square = tf.square(embedding_2)  # [batch_size,none,m,128]
        embeddings_2_square_sum = tf.reduce_sum(embeddings_2_square, 2)  # [batch_size,none,128]

        squared_sum_features_emb = embeddings_1_square_sum + embeddings_2_square_sum
        # ________ FM __________
        FM = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # [batch_size,None, 128]

        return FM

    def processUserTag(self, user_info, batch_size):
        userTags = user_info[0]
        userTags_w = user_info[1]
        userTags_k = user_info[2]

        user_embeddings = tf.nn.embedding_lookup(self.lookup_table, userTags)  # [batch_size,1,k,128]
        if userTags_w != None:  # weighted candidates' embeddings
            userTags_w = tf.reshape(userTags_w, [batch_size, 1, -1, 1])
            user_embeddings = tf.multiply(user_embeddings, userTags_w, name='user_embeddings_add_w')
        if userTags_k != None:  # instant,short,long,1,2,3
            userTags_positions_embeddings = self.position_embedding(userTags_k,
                                                                    self.num_userTagType,
                                                                    self.embedding_size,
                                                                    scope="userTags_positions_embeddings")
            # user_embeddings=tf.multiply(user_embeddings,userTags_positions_embeddings,name='user_embeddings_add_key')
            user_embeddings = tf.add(user_embeddings, userTags_positions_embeddings,
                                     name='user_embeddings_add_key')

        return user_embeddings

    def build_nfm_candidates_user_batch(self, newsTags, user_info, batch_size):
        with tf.variable_scope('nfm_weigths', reuse=tf.AUTO_REUSE):
            news_embeddings = tf.nn.embedding_lookup(self.lookup_table, newsTags)  # [batch_size,none,m,128]
            user_embeddings = self.processUserTag(user_info, batch_size)  # [batch_size,1,k,128]
            FM = self.build_nfm_batch(news_embeddings, user_embeddings)

        return FM

    ##############################################################################################################
    def build_nfm_cross_batch(self, embedding_1, embedding_2):
        # sum_userTag
        sum_embeddings_1 = tf.reduce_sum(embedding_1, 2)  # [b,1,128]
        # sum_newsTag
        sum_embeddings_2 = tf.reduce_sum(embedding_2, 2)  # [b,300,128]
        # sum_newsTag_userTag
        summed_features_emb = sum_embeddings_1 + sum_embeddings_2  # [b,300,128]
        # summed square of newsTag + userTag
        summed_features_emb_square = tf.square(summed_features_emb)  # [b,300,128]
        # summed square of userTag
        sum_embeddings_user_square = tf.square(sum_embeddings_1)
        # summed square of newsTag
        sum_embeddings_news_square = tf.square(sum_embeddings_2)
        # ________ FM __________
        FM = 0.5 * tf.subtract(summed_features_emb_square,
                               sum_embeddings_user_square + sum_embeddings_news_square)  # b,300,128
        return FM

    def build_nfm_candidates_user_cross_batch(self, newsTags, user_info, batch_size):
        with tf.variable_scope('nfm_weigths', reuse=tf.AUTO_REUSE):
            news_embeddings = tf.nn.embedding_lookup(self.lookup_table, newsTags)  # [b,300,10,128]
            user_embeddings = self.processUserTag(user_info, batch_size)  # [b,1,200,128]
            FM = self.build_nfm_cross_batch(news_embeddings, user_embeddings)

        return FM

    ###############################################################################################################
    def processInnerHistoryTags(self, history_tags, history_tags_w, batch_size):
        history_embeddings = tf.nn.embedding_lookup(self.lookup_table, history_tags)  # [batch_size,9,10,128]
        if history_tags_w != None:  # weighted candidates' embeddings
            history_embeddings = history_embeddings * history_tags_w

        history_embeddings = tf.reshape(history_embeddings, [batch_size, 1, -1, self.embedding_size],
                                        name='innerHistory_embeddings')
        return history_embeddings

    def build_nfm_candidates_innerHistory_batch(self, newsTags, history_tags, history_tags_w, batch_size):

        with tf.variable_scope('nfm_weigths', reuse=tf.AUTO_REUSE):
            news_embeddings = tf.nn.embedding_lookup(self.lookup_table, newsTags)  # [batch_size,none,m,128]
            history_embeddings = self.processInnerHistoryTags(history_tags, history_tags_w,
                                                              batch_size)  # [batch_size,9,10,128]
            FM = self.build_nfm_batch(news_embeddings, history_embeddings)

        return FM

    def build_nfm_candidates_innerHistory_cross_batch(self, newsTags, history_tags, history_tags_w, batch_size):

        with tf.variable_scope('nfm_weigths', reuse=tf.AUTO_REUSE):
            news_embeddings = tf.nn.embedding_lookup(self.lookup_table, newsTags)  # [batch_size,none,m,128]
            history_embeddings = self.processInnerHistoryTags(history_tags, history_tags_w,
                                                              batch_size)  # [batch_size,9,10,128]
            FM = self.build_nfm_cross_batch(news_embeddings, history_embeddings)

        return FM

    #############################################################################################################
    def processOuterHistoryTags(self, history, batch_size):
        # process history data
        h_newsTags = history[0]
        h_clicks = history[1]
        h_position = history[2]
        hist_embeddings = tf.nn.embedding_lookup(self.lookup_table, h_newsTags)
        hist_clicks_embeddings = self.position_embedding(h_clicks,
                                                         self.num_click,
                                                         self.embedding_size,
                                                         scope="hist_click_embedding")
        hist_positions_embeddings = self.position_embedding(h_position,
                                                            self.num_history,
                                                            self.embedding_size,
                                                            scope="hist_position_embedding")

        hist_embeddings = tf.multiply(hist_embeddings, hist_clicks_embeddings * hist_positions_embeddings,
                                      name='hist_embedding')
        hist_embeddings = tf.reshape(hist_embeddings, [batch_size, 1, -1, self.embedding_size],
                                     name='outerHistory_embeddings')
        return hist_embeddings

    def build_nfm_candidates_user_outerHistory_batch(self, newsTags, user_info, history, batch_size):
        with tf.variable_scope('nfm_weigths', reuse=tf.AUTO_REUSE):
            news_embeddings = tf.nn.embedding_lookup(self.lookup_table, newsTags)  # [batch_size,none,m,128]
            self.news_embeddings = news_embeddings
            user_embeddings = self.processUserTag(user_info, batch_size)  # [batch_size,1,k,128]
            hist_embeddings = self.processOuterHistoryTags(history, batch_size)

            user_hist_embeddings = tf.concat([user_embeddings, hist_embeddings], 2, name='user_add_history_embedding')
            # user_hist_embeddings = hist_embeddings
            FM = self.build_nfm_batch(news_embeddings, user_hist_embeddings)

        return FM

    def build_nfm_candidates_user_outerHistory_cross_batch(self, newsTags, user_info, history, batch_size):
        with tf.variable_scope('nfm_weigths', reuse=tf.AUTO_REUSE):
            news_embeddings = tf.nn.embedding_lookup(self.lookup_table, newsTags,
                                                     name='news_embeddings')  # [batch_size,none,m,128]
            user_embeddings = self.processUserTag(user_info, batch_size)  # [batch_size,1,k,128]
            hist_embeddings = self.processOuterHistoryTags(history, batch_size)

            user_hist_embeddings = tf.concat([user_embeddings, hist_embeddings], 2, name='user_add_history_embedding')
            FM = self.build_nfm_cross_batch(news_embeddings, user_hist_embeddings)

        return FM

    ##################################################################
    def transNFMVector(self, fm_embeddings, scope):
        with tf.variable_scope(scope):
            weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            fm_embeddings = tf.expand_dims(fm_embeddings, 1)  # [b,1,200,128]
            inter_1 = layers.conv2d(fm_embeddings, 256,
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

    def build_nfm_candidates_user_outerHistory_conbined(self, newsTags, user_info, history, batch_size):
        with tf.variable_scope('nfm_weigths', reuse=tf.AUTO_REUSE):
            news_embeddings = tf.nn.embedding_lookup(self.lookup_table, newsTags)  # [batch_size,none,m,128]
            user_embeddings = self.processUserTag(user_info, batch_size)  # [batch_size,1,k,128]
            hist_embeddings = self.processOuterHistoryTags(history, batch_size)
            user_hist_FM = self.build_nfm_batch(user_embeddings, hist_embeddings)
            user_hist_NFM = self.transNFMVector(user_hist_FM, scope='user_hist_FM_trans')

            user_candi_FM = self.build_nfm_batch(user_embeddings, news_embeddings)
            user_candi_NFM = self.transNFMVector(user_candi_FM, scope='user_candi_FM_trans')

            # user_hist_NFM_t = tf.tile(user_hist_NFM, [1, 300, 1])
            # res_conbined = tf.concat([user_hist_NFM_t, user_candi_NFM], axis=2, name='res_conbined')

            res_conbined = tf.add(user_hist_NFM, user_candi_NFM, name='res_conbined')

        return res_conbined

    def processOuterHistoryTags_mean(self, history, batch_size):
        # process history data
        h_newsTags = history[0]
        h_clicks = history[1]
        h_position = history[2]
        hist_embeddings = tf.nn.embedding_lookup(self.lookup_table, h_newsTags)
        hist_embeddings_mean = tf.reduce_mean(hist_embeddings, axis=2, keep_dims=True)

        hist_clicks_embeddings = self.position_embedding(h_clicks,
                                                         self.num_click,
                                                         self.embedding_size,
                                                         scope="hist_click_embedding")
        hist_positions_embeddings = self.position_embedding(h_position,
                                                            self.num_history,
                                                            self.embedding_size,
                                                            scope="hist_position_embedding")

        hist_embeddings = tf.multiply(hist_embeddings_mean, hist_clicks_embeddings + hist_positions_embeddings,
                                      name='hist_embedding')
        hist_embeddings = tf.reshape(hist_embeddings, [batch_size, 1, -1, self.embedding_size],
                                     name='outterHistory_embeddings')
        return hist_embeddings

    def build_nfm_candidates_user_outerHistory_conbined_fix(self, newsTags, user_info, history, batch_size):
        with tf.variable_scope('nfm_weigths', reuse=tf.AUTO_REUSE):
            news_embeddings = tf.nn.embedding_lookup(self.lookup_table, newsTags)  # [batch_size,none,m,128]
            user_embeddings = self.processUserTag(user_info, batch_size)  # [batch_size,1,k,128]

            hist_embeddings = self.processOuterHistoryTags_mean(history, batch_size)
            news_embeddings_mean = tf.reduce_mean(news_embeddings, axis=2, keep_dims=True)

            candi_hist_FM = self.build_nfm_batch(news_embeddings_mean, hist_embeddings)
            candi_hist_NFM = self.transNFMVector(candi_hist_FM, scope='candi_hist_FM_trans')

            candi_user_FM = self.build_nfm_batch(news_embeddings, user_embeddings)
            candi_user_NFM = self.transNFMVector(candi_user_FM, scope='user_candi_FM_trans')

            # user_hist_NFM_t = tf.tile(user_hist_NFM, [1, 300, 1])
            # res_conbined = tf.concat([candi_hist_NFM, candi_user_NFM], axis=2, name='res_conbined')

            res_conbined = tf.multiply(candi_hist_NFM, candi_user_NFM, name='res_conbined')

        return res_conbined

    ###################################################################################
    def nfm_transformer(self, queries, keys, scope='nfm_transformer'):
        '''
        :param queries: [b,n,t_q,c_q]
        :param keys: [b,m,t_k,c_k]
        :param num_units:
        :param scope:
        :return:
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            sum_keys = tf.reduce_sum(keys, axis=2, keep_dims=True)  # [b,m,1,c_k]
            cross_sum = tf.multiply(queries, sum_keys)  # [b,n,t_q,c_q]
            trans_cross_sum = tf.layers.dense(cross_sum, self.embedding_size)
        return trans_cross_sum

    def build_nfm_candidates_user_outerHistory_nfm_transformer(self, newsTags, user_info, history, batch_size):
        with tf.variable_scope('nfm_weigths', reuse=tf.AUTO_REUSE):
            news_embeddings = tf.nn.embedding_lookup(self.lookup_table, newsTags)  # [batch_size,none,m,128]
            user_embeddings = self.processUserTag(user_info, batch_size)  # [batch_size,1,k,128]
            hist_embeddings = self.processOuterHistoryTags(history, batch_size)

            user_hist_embeddings = tf.concat([user_embeddings, hist_embeddings], 2, name='user_add_history_embedding')

            fm_trans_0 = self.nfm_transformer(news_embeddings, user_hist_embeddings, scope='nfm_transformer_0')
            fm_trans_1 = self.nfm_transformer(fm_trans_0, user_hist_embeddings, scope='nfm_transformer_1')
            res = tf.reduce_sum(fm_trans_1, axis=2)
            # FM = self.build_nfm_batch(news_embeddings, user_hist_embeddings)

        return res

    ##########################################################################################
    def processOuterHistoryTagsBy_Click(self, history, batch_size):
        # process history data
        h_newsTags = history[0]
        h_clicks = history[1]
        h_position = history[2]
        hist_embeddings = tf.nn.embedding_lookup(self.lookup_table, h_newsTags)
        hist_clicks_embeddings = self.position_embedding(h_clicks,
                                                         self.num_click,
                                                         self.embedding_size,
                                                         scope="hist_click_embedding")
        hist_positions_embeddings = self.position_embedding(h_position,
                                                            self.num_history,
                                                            self.embedding_size,
                                                            scope="hist_position_embedding")

        hist_embeddings = tf.multiply(hist_embeddings, hist_clicks_embeddings * hist_positions_embeddings,
                                      name='hist_embedding')
        hist_embeddings = tf.reshape(hist_embeddings, [batch_size, 1, -1, self.embedding_size],
                                     name='outerHistory_embeddings')
        return hist_embeddings
