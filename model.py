import tensorflow as tf


class WordSegModel:
    def __init__(self, params, init_dict):
        self.params = params
        with tf.variable_scope('embeddings', regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_const'])):
            self.word_emb = tf.Variable(tf.random_uniform(
                [params['word_vocab_size'], params['word_emb_size']], minval=-0.1, maxval=0.1), name='word_emb')
            self.char_emb = tf.Variable(tf.random_uniform(
                [params['char_vocab_size'], params['char_emb_size']], minval=-0.1, maxval=0.1), name='char_emb')
            self.bigram_emb = tf.Variable(tf.random_uniform(
                [params['bigram_vocab_size'], params['bigram_emb_size']], minval=-0.1, maxval=0.1), name='action_emb')
            self.action_emb = tf.Variable(tf.random_uniform(
                [params['action_size'], params['action_emb_size']], minval=-0.1, maxval=0.1), name='action_emb')
        
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.beam_size = tf.Variable(20, dtype=tf.int32, trainable=False, name='beam_size_static')
        self.define_placeholder()
        self.build_graph()
        
    
    def define_placeholder(self):
        with tf.variable_scope('placeholder'):
            self.emb_pl = dict()
            self.emb_pl['word'] = tf.placeholder(tf.float32, name='word_emb_placeholder')
            self.emb_pl['char'] = tf.placeholder(tf.float32, name='char_emb_placeholder')

            self.x = dict()
            self.x['buffer'] = tf.placeholder(tf.int32, [None, None], name='buffer_placeholder')
            self.x['stack'] = tf.placeholder(tf.int32, [None, None], name='stack_placeholder')
            self.x['actions'] = tf.placeholder(tf.int32, [None, None], name='actions_placeholder')
            self.x['bigram'] = tf.placeholder(tf.int32, [None, None], name='bigram_placeholder')
            self.x['buffer_len'] = tf.placeholder(tf.int32, [None], name='buffer_len')
            self.x['buffer_fwd_len'] = tf.placeholder(tf.int32, [None], name='buffer_fwd_len')
            self.x['buffer_bwd_len'] = tf.placeholder(tf.int32, [None], name='buffer_bwd_len')
            self.x['stack_len'] = tf.placeholder(tf.int32, [None], name='stack_len')
            self.x['actions_len'] = tf.placeholder(tf.int32, [None], name='actions_len')
            self.buffer_char = tf.placeholder(tf.string, [None], name='buffer_char_placeholder')
            
            self.labels = tf.placeholder(tf.int32, [None], name='labels_placeholder')
            self.gold_sequence = tf.placeholder(tf.int32, [None], name='gold_sequence_placeholder')
    
    def build_graph(self):
        self.drop_embeddings('TRAIN')
        self.convolution_layer(self.x['stack'], self.x['actions'], self.x['buffer'], self.x['bigram'])
        self.recurrent_layer(self.x['stack_len'], self.x['actions_len'], self.x['buffer_len'], self.x['buffer_fwd_len'], self.x['buffer_fwd_len'])
        self.hidden_layer('TRAIN')
        self.output_layer()
        
        self.regularization_loss()
        
        self.simple_loss_fn()
        self.simple_optimizer = tf.train.AdamOptimizer().minimize(self.simple_loss)
    
    def init_embeddings(self):
        init_word = tf.assign(self.word_emb, self.emb_pl['word'])
        init_char = tf.assign(self.char_emb, self.emb_pl['char'])
        return [init_word, init_char]
    
    def increase_global_step(self):
        return tf.assign_add(self.global_step, 1)
    
    def assign_beam_size(self, beam_size):
        return tf.assign(self.beam_size, beam_size)
    
    def drop_embeddings(self, mode):
        with tf.variable_scope('embedding_dropout', reuse=tf.AUTO_REUSE):
            if mode == 'TRAIN':
                self.dropped_word_emb = tf.nn.dropout(self.word_emb, self.params['emb_dropout_keep_rate'])
                self.dropped_action_emb = tf.nn.dropout(self.action_emb, self.params['emb_dropout_keep_rate'])
                self.dropped_char_emb = tf.nn.dropout(self.char_emb, self.params['emb_dropout_keep_rate'])
                self.dropped_bigram_emb = tf.nn.dropout(self.bigram_emb, self.params['emb_dropout_keep_rate'])
            else:
                self.dropped_word_emb = self.word_emb
                self.dropped_action_emb = self.action_emb
                self.dropped_char_emb = self.char_emb
                self.dropped_bigram_emb = self.bigram_emb
    
    def convolution_layer(self, stack, actions, buffer, bigram):
        """
        Input:
        1) a sequence of words in stack (w1, ..., wn)
        2) a sequence of characters in buffer (c1, ..., cm)
        3) a sequence of bigram in buffer (b1, ..., bm-1)
        4) a sequence of previous actions (a1, ..., at)
        Process:
        1. Lookup with embeddings
        2. Use CNN described part 3.1 (input representation) in the paper
        Return:
        x^w, x^a, x^c representing a sequence of (represented) words, actions, and characters respectively
        -
        """
        def convolute(data_seq, window_size, emb_size, no_output, scope_name):
            with tf.variable_scope('represent_' + scope_name, reuse=tf.AUTO_REUSE,
                                   regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_const'])):
                weights = tf.get_variable('w_conv_' + scope_name,
                                          shape=[window_size, emb_size, 1, no_output])
                expanded_data = tf.expand_dims(data_seq, axis=-1)
                conv = tf.nn.conv2d(expanded_data, weights, [1, 1, 1, 1], padding='VALID')

                bias = tf.get_variable('b_conv_' + scope_name, shape=[1, no_output],
                                       initializer=tf.zeros_initializer())
                rep_sum = tf.reshape(conv, [tf.shape(conv)[0], tf.shape(conv)[1], -1]) + bias
                rep = tf.tanh(rep_sum)
                return rep

        with tf.variable_scope('embedding_lookup', reuse=tf.AUTO_REUSE):
            stack = tf.nn.embedding_lookup(self.dropped_word_emb, stack)
            prev_actions = tf.nn.embedding_lookup(self.dropped_action_emb, actions)
            buff = tf.nn.embedding_lookup(self.dropped_char_emb, buffer)
            bigram_buff = tf.nn.embedding_lookup(self.dropped_bigram_emb, bigram)
            concat_buffer = tf.concat([buff, bigram_buff], axis=-1)

        self.x_w = convolute(stack, window_size=2, emb_size=self.params['word_emb_size'],
                        no_output=self.params['rep_stack'], scope_name='stack')
        self.x_a = convolute(prev_actions, window_size=2, emb_size=self.params['action_emb_size'],
                        no_output=self.params['rep_actions'], scope_name='actions')
        concat_buffer_emb_size = self.params['char_emb_size'] + self.params['bigram_emb_size']
        self.x_c = convolute(concat_buffer, window_size=5, emb_size=concat_buffer_emb_size,
                        no_output=self.params['rep_buffer'], scope_name='buffer')
        return self.x_w, self.x_a, self.x_c

    def recurrent_layer(self, stack_len, actions_len, buffer_len, buffer_fwd_len, buffer_bwd_len):
        def apply_recurrent(sequence, length, state_size, scope_name):
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE,
                                   regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_const'])) as rnn_scope:
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
                outputs, state = tf.nn.dynamic_rnn(rnn_cell, sequence, length, scope=rnn_scope, dtype=tf.float32)

                batch_range = tf.range(tf.shape(sequence)[0])
                index = tf.stack([batch_range, (length-1)], axis=-1)
                return tf.gather_nd(outputs, index)

        self.r_w = apply_recurrent(self.x_w, stack_len, self.params['lstm_word'], 'recurrent_word')
        self.r_a = apply_recurrent(self.x_a, actions_len, self.params['lstm_actions'], 'recurrent_actions')

        # bi-lstm on buffer
        forward_buff = apply_recurrent(self.x_c, buffer_fwd_len,
                                       self.params['lstm_fwd_char'], 'recurrent_fwd_char')
        reverse_buffer = tf.reverse_sequence(self.x_c, buffer_len, seq_axis=1)
        backward_buff = apply_recurrent(reverse_buffer, buffer_bwd_len,
                                        self.params['lstm_bwd_char'], 'recurrent_bwd_char')
        self.r_c = tf.concat([forward_buff, backward_buff], axis=-1)
        return self.r_w, self.r_a, self.r_c

    def hidden_layer(self, run_mode):
        with tf.variable_scope('hidden_layer', reuse=tf.AUTO_REUSE,
                               regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_const'])):
            r_sep = tf.concat([self.r_w, self.r_c, self.r_a], axis=-1)
            h_sep = tf.layers.dense(r_sep, self.params['hidden_sep'], activation=tf.tanh)
            if run_mode == 'TRAIN':
                self.h_sep = tf.nn.dropout(h_sep, self.params['simple_dropout_keep_rate'])
            else:
                self.h_sep = h_sep

            r_app = tf.concat([self.r_c, self.r_a], axis=-1)
            h_app = tf.layers.dense(r_app, self.params['hidden_app'], activation=tf.tanh)
            if run_mode == 'TRAIN':
                self.h_app = tf.nn.dropout(h_app, self.params['simple_dropout_keep_rate'])
            else:
                self.h_app = h_app

            return self.h_sep, self.h_app

    def output_layer(self):
        with tf.variable_scope('output_layer', reuse=tf.AUTO_REUSE,
                               regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_const'])):
            w_sep = tf.get_variable('weight_sep', shape=[1, self.params['hidden_sep']],
                                    initializer=tf.contrib.layers.xavier_initializer())
            self.score_sep = tf.reduce_sum(tf.multiply(w_sep, self.h_sep), axis=1)

            w_app = tf.get_variable('weight_app', shape=[1, self.params['hidden_app']],
                                    initializer=tf.contrib.layers.xavier_initializer())
            self.score_app = tf.reduce_sum(tf.multiply(w_app, self.h_app), axis=1)

            return self.score_sep, self.score_app
    
    def simple_loss_fn(self):
        with tf.name_scope('simple_loss'):
            logit = tf.transpose(tf.stack([self.score_sep, self.score_app], axis=0))
            one_hot_labels = tf.one_hot(self.labels - 2, 2)
            self.simple_loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=one_hot_labels))
            return self.simple_loss

    def regularization_loss(self):
        with tf.name_scope('regularization_loss'):
            self.reg_loss = tf.losses.get_regularization_losses()
        return self.reg_loss
    
    def run_basic(self, run_mode):
        """
        Run through the network using placeholder and run simple_loss_fn
        """
        self.convolution_layer(self.x['stack'], self.x['actions'], self.x['buffer'], self.x['bigram'])
        self.recurrent_layer(self.x['stack_len'], self.x['actions_len'], self.x['buffer_len'], self.x['buffer_fwd_len'], self.x['buffer_bwd_len'])
        self.hidden_layer(run_mode)
        score_sep, score_app = self.output_layer()
        loss = self.simple_loss_fn()
        return [loss, score_sep, score_app]
