import tensorflow as tf


class WordSegModel:
    def __init__(self, params):
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
        self.define_placeholder()
        self.define_storing_variables()
        self.build_graph()

        self.optimizer = tf.train.AdamOptimizer()
        self.simple_optimizer = tf.train.AdamOptimizer().minimize(self.simple_loss)
        self.reset_accumulated_gradients()
        self.apply_gradients()

    def define_placeholder(self):
        with tf.variable_scope('placeholder'):
            self.emb_pl = dict()
            self.emb_pl['word'] = tf.placeholder(tf.float32, name='word_emb_placeholder')
            self.emb_pl['char'] = tf.placeholder(tf.float32, name='char_emb_placeholder')

            self.input = dict()
            self.input['buffer'] = tf.placeholder(tf.int32, [None, None], name='buffer_placeholder')
            self.input['stack'] = tf.placeholder(tf.int32, [None, None], name='stack_placeholder')
            self.input['actions'] = tf.placeholder(tf.int32, [None, None], name='actions_placeholder')
            self.input['bigram'] = tf.placeholder(tf.int32, [None, None], name='bigram_placeholder')
            self.input['buffer_len'] = tf.placeholder(tf.int32, [None], name='buffer_len')
            self.input['buffer_fwd_len'] = tf.placeholder(tf.int32, [None], name='buffer_fwd_len')
            self.input['buffer_bwd_len'] = tf.placeholder(tf.int32, [None], name='buffer_bwd_len')
            self.input['stack_len'] = tf.placeholder(tf.int32, [None], name='stack_len')
            self.input['actions_len'] = tf.placeholder(tf.int32, [None], name='actions_len')

            self.labels = tf.placeholder(tf.int32, [None], name='labels_placeholder')
            self.mask = tf.placeholder(tf.float32, [None], name='mask_placeholder')
            self.lookup_key_pl = tf.placeholder(tf.string, [None], name='lookup_keys_placeholder')
            self.lookup_val_pl = tf.placeholder(tf.int32, [None], name='lookup_values_placeholder')
    
    def define_storing_variables(self):
        with tf.variable_scope('storing_variables'):
            # self.v = dict()
            # for input_name in self.input:
            #     self.v[input_name] = tf.get_variable(input_name + '_static', shape=[1, 1], trainable=False, dtype=tf.int32, validate_shape=False)
            # self.acc_score = tf.get_variable('score_static', shape=[1], trainable=False, dtype=tf.float32)
            self.beam_size = tf.get_variable('beam_size_static', shape=[], trainable=False, dtype=tf.int32)
            self.padded_stack = tf.Variable([[0, 0]], dtype=tf.int32, trainable=False, name='padded_stack_static', validate_shape=False)
            self.gold_score = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='gold_score_static')
    
#     def assign_from_beam(self, need_to_pad, pad_key=None):
#         """
#         Edit input item and score list according to the pruned beam (also check for eligibility, e.g. APP to the empty stack)
#         """
#         with tf.name_scope('assign_from_beam'):
#             ops = list()
#             next_item = dict()
#             assign_ops = dict()
#             batch_size = tf.shape(self.next_stack_len)[0]
#             next_item['buffer'] = tf.tile(self.v['buffer'][0:1,:], [batch_size, 1])
#             next_item['bigram'] = tf.tile(self.v['bigram'][0:1,:], [batch_size, 1])
#             next_item['buffer_len'] = tf.tile([self.v['buffer_len'][0]], [batch_size])
#             next_item['buffer_fwd_len'] = tf.tile([self.v['buffer_fwd_len'][0] + 1], [batch_size])
#             next_item['buffer_bwd_len'] = tf.tile([self.v['buffer_bwd_len'][0] - 1], [batch_size])
#             next_item['stack_len'] = self.next_stack_len - 1
#             next_item['actions_len'] = tf.tile([self.v['actions_len'][0] + 1], [batch_size])
            
#             # actions
#             old_actions = tf.concat([tf.concat([self.v['actions'][1:], tf.tile([[2]], [tf.shape(self.v['actions'])[0] - 1, 1])], axis=-1),
#                                     tf.concat([self.v['actions'][1:], tf.tile([[3]], [tf.shape(self.v['actions'])[0] - 1, 1])], axis=-1)],
#                                    axis=0)
#             sorted_old_actions = tf.gather(old_actions, self.new_sorted_index)
#             next_gold_action = tf.concat([self.v['actions'][0:1,:], [[self.next_action]]], axis=-1)
#             next_item['actions'] = tf.concat([next_gold_action, sorted_old_actions], axis=0)

#             # stack
#             old_stack = tf.gather(tf.tile(self.v['stack'][1:], [2, 1]), self.new_sorted_index)
#             old_stack_with_gold = tf.concat([self.v['stack'][0:1,:], old_stack], axis=0)
#             if need_to_pad:
#                 padded_stack = tf.concat([old_stack_with_gold, tf.tile([[pad_key]], [batch_size, 1])], axis=-1)
#             else:
#                 padded_stack = old_stack_with_gold
#             ops.append(tf.assign(self.padded_stack, padded_stack, validate_shape=False))  # must assign before doing scatter_nd

#             for input_name in next_item:
#                 ops.append(tf.assign(self.v[input_name], next_item[input_name], validate_shape=False))
            
#             # score
#             new_score_list = tf.concat([[self.gold_score], tf.gather(self.acc_score, self.new_sorted_index)], axis=-1)
#             ops.append(tf.assign(self.acc_score, new_score_list, validate_shape=False))
#             return ops
    
#     def update_stack(self):
#         """
#         An extension method from assign_from_beam to edit a problem on editing scatter_nd_update (mutable tensor and shape validation)
#         """
#         batch_size = tf.shape(self.next_stack)[0]
#         updated_indices = tf.stack([tf.range(0, batch_size, 1), (self.next_stack_len-1)], axis=-1)
#         assign_nd = tf.scatter_nd_update(self.padded_stack, updated_indices, self.next_stack)
#         update_op = tf.assign(self.v['stack'], assign_nd, validate_shape=False)
        
#         return update_op

    def build_graph(self):
        self.convolution_layer()
        self.recurrent_layer()
        self.hidden_layer()
        self.output_layer()
        self.regularization_loss()
        self.simple_loss_fn()
        self.max_margin_loss_fn()
        self.word_emb_lookup = None

    def init_model(self):
        self.init = tf.global_variables_initializer()
        return self.init

    def init_embeddings(self):
        init_word = tf.assign(self.word_emb, self.emb_pl['word'])
        init_char = tf.assign(self.char_emb, self.emb_pl['char'])
        return [init_word, init_char]
    
    def increase_global_step(self):
        assign_op = tf.assign_add(self.global_step, 1)
        return assign_op

    def assign_beam_size(self, beam_size):
        assign_op = tf.assign(self.beam_size, beam_size)
        return assign_op

    def reset_accumulated_gradients(self):
        with tf.name_scope('init_accumulated_gradients'):
            self.trainable_vars = tf.trainable_variables()
            self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
                               for tv in self.trainable_vars]
            init_accum_vars = [tf.assign(tv, tf.zeros_like(tv)) for tv in self.accum_vars]
            return init_accum_vars

    def convolution_layer(self):
        """
        Input:
        1) a sequence of words in stack (w1, ..., wn)
        2) a sequence of characters in buffer (c1, ..., cm)
        3) a sequence of previous actions (a1, ..., at)
        Process:
        1. Lookup with embeddings
        2. Use CNN described part 3.1 (input representation) in the paper
        Return:
        x^w, x^a, x^c representing a sequence of (represented) words, actions, and characters respectively
        -
        """
        def convolute(data_seq, window_size, emb_size, no_output, scope_name):
            with tf.variable_scope('represent_' + scope_name, reuse=None,
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

        with tf.variable_scope('embedding_lookup', reuse=None):
            stack = tf.nn.embedding_lookup(self.word_emb, self.v['stack'])
            prev_actions = tf.nn.embedding_lookup(self.action_emb, self.v['actions'])
            buff = tf.nn.embedding_lookup(self.char_emb, self.v['buffer'])
            bigram_buff = tf.nn.embedding_lookup(self.bigram_emb, self.v['bigram'])
            concat_buffer = tf.concat([buff, bigram_buff], axis=-1)

        self.x_w = convolute(stack, window_size=2, emb_size=self.params['word_emb_size'],
                        no_output=self.params['rep_stack'], scope_name='stack')
        self.x_a = convolute(prev_actions, window_size=2, emb_size=self.params['action_emb_size'],
                        no_output=self.params['rep_actions'], scope_name='actions')
        concat_buffer_emb_size = self.params['char_emb_size'] + self.params['bigram_emb_size']
        self.x_c = convolute(concat_buffer, window_size=5, emb_size=concat_buffer_emb_size,
                        no_output=self.params['rep_buffer'], scope_name='buffer')
        return self.x_w, self.x_a, self.x_c

    def recurrent_layer(self):
        def apply_recurrent(sequence, length, state_size, scope_name):
            with tf.variable_scope(scope_name, reuse=None,
                                   regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_const'])) as rnn_scope:
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
                outputs, state = tf.nn.dynamic_rnn(rnn_cell, sequence, length, scope=rnn_scope, dtype=tf.float32)

                batch_range = tf.range(tf.shape(sequence)[0])
                index = tf.stack([batch_range, (length-1)], axis=-1)
                return tf.gather_nd(outputs, index)

        self.r_w = apply_recurrent(self.x_w, self.v['stack_len'], self.params['lstm_word'], 'recurrent_word')
        self.r_a = apply_recurrent(self.x_a, self.v['actions_len'], self.params['lstm_actions'], 'recurrent_actions')

        # bi-lstm on buffer
        forward_buff = apply_recurrent(self.x_c, self.v['buffer_fwd_len'],
                                       self.params['lstm_fwd_char'], 'recurrent_fwd_char')
        reverse_buffer = tf.reverse_sequence(self.x_c, self.v['buffer_len'], seq_axis=1)
        backward_buff = apply_recurrent(reverse_buffer, self.v['buffer_bwd_len'],
                                        self.params['lstm_bwd_char'], 'recurrent_bwd_char')
        self.r_c = tf.concat([forward_buff, backward_buff], axis=-1)
        return self.r_w, self.r_a, self.r_c

    def hidden_layer(self):
        with tf.variable_scope('hidden_layer', reuse=None,
                               regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_const'])):
            r_sep = tf.concat([self.r_w, self.r_c, self.r_a], axis=-1)
            self.h_sep = tf.layers.dense(r_sep, self.params['hidden_sep'], activation=tf.tanh)
            # self.h_sep = tf.nn.dropout(h_sep, self.params['dropout_keep_rate'])

            r_app = tf.concat([self.r_c, self.r_a], axis=-1)
            self.h_app = tf.layers.dense(r_app, self.params['hidden_app'], activation=tf.tanh)
            # self.h_app = tf.nn.dropout(h_app, self.params['dropout_keep_rate'])

            return self.h_sep, self.h_app

    def output_layer(self):
        with tf.variable_scope('output_layer', reuse=None,
                               regularizer=tf.contrib.layers.l2_regularizer(self.params['reg_const'])):
            w_sep = tf.get_variable('weight_sep', shape=[1, self.params['hidden_sep']],
                                    initializer=tf.contrib.layers.xavier_initializer())
            self.score_sep = tf.reduce_sum(tf.multiply(w_sep, self.h_sep), axis=1)

            w_app = tf.get_variable('weight_app', shape=[1, self.params['hidden_app']],
                                    initializer=tf.contrib.layers.xavier_initializer())
            self.score_app = tf.reduce_sum(tf.multiply(w_app, self.h_app), axis=1)

            return self.score_sep, self.score_app
    
    def prune_beam(self):
        """
        After getting score_sep and score_app, sum with the existing self.acc_score and then sort.
        We will consider only score[1:] because score[0] represents score for the gold state
        """
        # store last gold score
        gold_seq_score = tf.stack([self.score_sep[0], self.score_app[0]])
        gold_score = tf.reduce_sum((tf.multiply(gold_seq_score, self.mask)))
        acc_gold_score = self.acc_score[0] + gold_score
        assign_gold_score = tf.assign(self.gold_score, acc_gold_score)
        
        # calculate new score
        tiled_score = tf.tile(self.acc_score[1:], [2])
        new_score = tiled_score + tf.concat([self.score_sep[1:], self.score_app[1:]], axis=-1)
        assign_new_score = tf.assign(self.acc_score, new_score, validate_shape=False)
        
        # prune beam
        _, sorted_index = tf.nn.top_k(new_score, 
                                                 tf.minimum(self.beam_size, tf.shape(self.v['stack'])[0]))
        return sorted_index, assign_new_score, assign_gold_score

    def simple_loss_fn(self):
        with tf.name_scope('simple_loss'):
            logit = tf.transpose(tf.stack([self.score_sep, self.score_app], axis=0))
            one_hot_labels = tf.one_hot(self.labels - 2, 2)
            self.simple_loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=one_hot_labels))
            return self.simple_loss + self.regularization_loss()

    def max_margin_loss_fn(self):
        with tf.name_scope('max_margin_loss'):
            gold_seq_score = tf.stack([self.score_sep[0], self.score_app[0]])
            gold_score = tf.reduce_sum((tf.multiply(gold_seq_score, self.mask)))

            identity = tf.reshape(tf.tile(tf.reshape(tf.reverse(self.mask, axis=[0]), [2, 1]),
                                          [1, tf.shape(self.score_app)[0] - 1]), [-1])
            all_output = tf.concat([self.score_sep[1:], self.score_app[1:]], axis=-1)
            max_predicted = tf.reduce_max(all_output + tf.multiply(self.params['margin'], identity))
            self.max_margin_loss = tf.subtract(max_predicted, gold_score)
            self.added_max_margin_loss = self.max_margin_loss + self.regularization_loss()

            return self.added_max_margin_loss

    def regularization_loss(self):
        with tf.name_scope('regularization_loss'):
            self.reg_loss = tf.losses.get_regularization_losses()
            return self.reg_loss

    def accumulate_gradient(self, loss_fn):
        with tf.name_scope('accumulate_gradients'):
            gradient_vars = self.optimizer.compute_gradients(loss_fn(), self.trainable_vars)
            self.accumulate_ops = [tf.assign_add(self.accum_vars[i], gradient[0]) for i, gradient in enumerate(gradient_vars)]
            return self.accumulate_ops

    def average_gradients(self):
        with tf.name_scope('average_gradients'):
            ops = list()
            for i, tv in enumerate(self.accum_vars):
                assign_op = tf.assign(self.accum_vars[i], tf.divide(tv, self.no_steps))
                ops.append(assign_op)

            return ops

    def apply_gradients(self):
        with tf.name_scope('apply_gradients'):
            self.train_step = self.optimizer.apply_gradients(
                [(self.accum_vars[i], var) for i, var in enumerate(self.trainable_vars)],
                global_step=self.global_step
            )
            return self.train_step

