import tensorflow as tf


class BeamWordSegModel:
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
        lookup_key, lookup_val = init_dict['word_emb']
        self.init_lookup_table(lookup_key, lookup_val)
        self.build_graph()
        self.define_gradient_vars()
    
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
        self.convolution_layer(self.x['stack'], self.x['actions'], self.x['buffer'], self.x['bigram'])
        self.recurrent_layer(self.x['stack_len'], self.x['actions_len'], self.x['buffer_len'], self.x['buffer_fwd_len'], self.x['buffer_fwd_len'])
        self.hidden_layer('TRAIN')
        self.output_layer()
        
        self.regularization_loss()
        
        self.simple_loss_fn()
        self.define_output_dict()
    
    def init_lookup_table(self, lookup_key, lookup_val):
        self.word_emb_lookup = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(lookup_key, lookup_val, key_dtype=tf.string, value_dtype=tf.int32)
            , 0)
    
    def increase_global_step(self):
        return tf.assign_add(self.global_step, 1)
    
    def init_embeddings(self):
        init_word = tf.assign(self.word_emb, self.emb_pl['word'])
        init_char = tf.assign(self.char_emb, self.emb_pl['char'])
        return [init_word, init_char]
    
    def assign_beam_size(self, beam_size):
        return tf.assign(self.beam_size, beam_size)
    
    def define_gradient_vars(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'])
        self.tvs = tf.trainable_variables()
        self.grad_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in self.tvs]
        self.accu_grad_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in self.tvs]
        grad = self.optimizer.compute_gradients(self.trained_beam_loss, self.tvs)
        self.grad_assign_ops = [tf.assign(self.grad_vars[i], gv[0]) for i, gv in enumerate(grad)]
        self.reset_accu_grad_ops = [tf.assign(self.accu_grad_vars[i], tf.zeros_like(self.accu_grad_vars[i])) for i in range(len(self.accu_grad_vars))]
        self.accu_grad_assign_ops = [tf.assign_add(self.accu_grad_vars[i], self.grad_vars[i]) for i in range(len(self.grad_vars))]
        self.beam_optimize = self.optimizer.apply_gradients(zip(self.accu_grad_vars, self.tvs))
    
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
            stack = tf.nn.embedding_lookup(self.word_emb, stack)
            prev_actions = tf.nn.embedding_lookup(self.action_emb, actions)
            buff = tf.nn.embedding_lookup(self.char_emb, buffer)
            bigram_buff = tf.nn.embedding_lookup(self.bigram_emb, bigram)
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
                self.h_sep = tf.nn.dropout(h_sep, self.params['beam_dropout_keep_rate'])
            else:
                self.h_sep = h_sep

            r_app = tf.concat([self.r_c, self.r_a], axis=-1)
            h_app = tf.layers.dense(r_app, self.params['hidden_app'], activation=tf.tanh)
            if run_mode == 'TRAIN':
                self.h_app = tf.nn.dropout(h_app, self.params['beam_dropout_keep_rate'])
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
        self.reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.params['reg_const']
        return self.reg_loss
    
    def run_beam(self, run_mode):
        """
        Run beam search for a sentence
        """
        var_list = ['score', 'stack_char', 'buffer_char', 'gold_seq', 'loss', 'buffer', 'stack', 'actions', 'bigram', 'buffer_len', 'buffer_fwd_len',
                   'buffer_bwd_len', 'stack_len', 'actions_len', 'run_mode']
        tensor_shape = dict(
            [(key, tf.TensorShape([None])) for key in ['score', 'stack_char', 'buffer_char', 'gold_seq', 'buffer_len', 
                                                       'buffer_fwd_len', 'buffer_bwd_len', 'stack_len', 'actions_len', 'loss']]
            + [(key, tf.TensorShape([None, None])) for key in ['buffer', 'stack', 'actions', 'bigram']]
            + [('run_mode', tf.TensorShape([]))])
        pad_index = 3
        
        def loop_body(*args):
            loop_var = dict()
            concat_var = dict()
            modified = dict()
            
            for i, var_name in enumerate(var_list):
                loop_var[var_name] = args[i]

            self.convolution_layer(loop_var['stack'], loop_var['actions'], loop_var['buffer'], loop_var['bigram'])
            self.recurrent_layer(loop_var['stack_len'], loop_var['actions_len'], loop_var['buffer_len'], loop_var['buffer_fwd_len'], 
                                 loop_var['buffer_bwd_len'])
            self.hidden_layer(run_mode)
            score_sep, score_app = self.output_layer()
            
            with tf.variable_scope('inner_loop_beam_predictions', reuse=tf.AUTO_REUSE):
                # Content in stacks and actions are differed with actions SEP/APP and for the first time, only 'SEP' action can be done
                def sort_first_time():
                    score = loop_var['score'][1:] + score_sep[1:]
                    stack_char = tf.expand_dims(loop_var['buffer_char'][0], axis=0)
                    stack = tf.concat([loop_var['stack'][1:, :], [[pad_index, pad_index]]], axis=-1)
                    stack_len = tf.constant([2, 2], dtype=tf.int32)
                    actions = tf.concat([loop_var['actions'][1:], [[2]]], axis=-1)
                    return score, stack_char, stack_len, actions
                
                def sort_regular():
                    prediction_len = tf.shape(score_sep[1:])[0]
                    score_gold_sep = lambda: tf.tile(loop_var['score'][1:], [2]) + tf.concat([score_sep[1:], score_app[1:] + self.params['margin']], axis=-1)
                    score_gold_app = lambda: tf.tile(loop_var['score'][1:], [2]) + tf.concat([score_sep[1:] + self.params['margin'], score_app[1:]], axis=-1)
                    score = tf.cond(tf.equal(loop_var['gold_seq'][0], 2), score_gold_sep, score_gold_app)
                    # score = tf.tile(loop_var['score'][1:], [2]) + tf.concat([score_sep[1:], score_app[1:]], axis=-1)
                    
                    new_coming_chars = tf.tile([loop_var['buffer_char'][0]], [prediction_len])
                    stack_char_sep = new_coming_chars
                    stack_char_app = loop_var['stack_char'][1:] + new_coming_chars
                    stack_char = tf.concat([stack_char_sep, stack_char_app], axis=0)
                    
                    stack_len = tf.concat([(loop_var['stack_len'][1:] + 1), loop_var['stack_len'][1:]], axis=0)
                    
                    actions = tf.concat([tf.concat([loop_var['actions'][1:], tf.tile([[2]], [prediction_len, 1])], axis=-1),
                                        tf.concat([loop_var['actions'][1:], tf.tile([[3]], [prediction_len, 1])], axis=-1)], axis=0)
                    return score, stack_char, stack_len, actions
                    
                allseq = dict()
                allseq['score'], allseq['stack_char'], allseq['stack_len'], allseq['actions'] = tf.cond(
                    tf.equal(loop_var['buffer_fwd_len'][0], 1), sort_first_time, sort_regular)
                
                
                # sort sequences of actions according to the score
                _, sorted_index = tf.nn.top_k(allseq['score'], tf.minimum(self.beam_size, tf.shape(allseq['score'])[0]))
                predicted = dict()
                new_beam_len = tf.shape(sorted_index)[0]
                # new_beam_len = tf.Print(new_beam_len, [loop_var['actions'], loop_var['score'], score_sep, score_app], 'debug', summarize=200)
                predicted['buffer'] = tf.tile(loop_var['buffer'][0:1, :], [new_beam_len, 1])
                predicted['bigram'] = tf.tile(loop_var['bigram'][0:1, :], [new_beam_len, 1])
                predicted['actions_len']  = tf.tile([loop_var['actions_len'][0] + 1], [new_beam_len])
                predicted['buffer_len'] = tf.tile([loop_var['buffer_len'][0]], [new_beam_len])
                predicted['buffer_fwd_len'] = tf.tile([loop_var['buffer_fwd_len'][0] + 1], [new_beam_len])
                predicted['buffer_bwd_len'] = tf.tile([loop_var['buffer_bwd_len'][0] - 1], [new_beam_len])
                
                for key in allseq:
                    predicted[key] = tf.gather(allseq[key], sorted_index)
            
            # take action on the gold sequence (first row on every tensor in var_loop)
            with tf.variable_scope('inner_loop_beam_gold', reuse=tf.AUTO_REUSE):
                gold = dict()
                def perform_sep_on_gold():
                    top_score = [loop_var['score'][0] + score_sep[0]]
                    stack_char = [loop_var['buffer_char'][0]]
                    stack_len = [loop_var['stack_len'][0] + 1]
                    actions = tf.concat([[loop_var['actions'][0]], [[2]]], axis=-1)
                    return top_score, stack_char, stack_len, actions
                
                def perform_app_on_gold():
                    top_score = [loop_var['score'][0] + score_app[0]]
                    stack_char = [loop_var['stack_char'][0] + loop_var['buffer_char'][0]]
                    stack_len = [loop_var['stack_len'][0]]
                    actions = tf.concat([[loop_var['actions'][0]], [[3]]], axis=-1)
                    return top_score, stack_char, stack_len, actions
                
                gold['score'], gold['stack_char'], gold['stack_len'], gold['actions'] = tf.cond(tf.equal(loop_var['gold_seq'][0], 2),
                                                                                            perform_sep_on_gold, perform_app_on_gold)
                for key in ['buffer', 'bigram', 'actions_len', 'buffer_len', 'buffer_fwd_len', 'buffer_bwd_len']:
                    gold[key] = [predicted[key][0]]
            
            # combine gold sentence and prediction + re-evaluate stack tensor
            with tf.variable_scope('inner_loop_beam_combine', reuse=tf.AUTO_REUSE):
                for key in predicted:
                    modified[key] = tf.concat([gold[key], predicted[key]], axis=0)

                # make a new stack
                extended_stack = tf.concat([[loop_var['stack'][0]],
                                           tf.gather(tf.tile(loop_var['stack'][1:], [2, 1]), sorted_index)], axis=0)
                add_new_col_to_stack = lambda: tf.concat([extended_stack, tf.tile([[pad_index]], [tf.shape(extended_stack)[0], 1])], axis=-1)
                use_same_stack = lambda: extended_stack
                compare_op = tf.greater_equal(tf.reduce_max(modified['stack_len']), tf.shape(loop_var['stack'])[1])
                stack = tf.cond(compare_op, add_new_col_to_stack, use_same_stack)
                
                # fill new value
                beam_len = tf.shape(modified['buffer'])[0]
                index_range = tf.range(0, beam_len)
                sparse_indices = tf.stack([index_range, modified['stack_len']], axis=1)
                sparse_values = self.word_emb_lookup.lookup(modified['stack_char'])
                dummy_values = tf.tile([-1], [beam_len])
                
                new_tensor = tf.sparse_to_dense(sparse_indices, [beam_len, tf.shape(stack)[1]], sparse_values)
                mask_tensor = tf.sparse_to_dense(sparse_indices, [beam_len, tf.shape(stack)[1]], dummy_values) + 1
                modified['stack'] = tf.multiply(stack, mask_tensor) + new_tensor
            
            # find max margin loss for each iteration
            with tf.variable_scope('inner_loop_margin_loss'):
                gold_score = modified['score'][0]
                tiled_gold_action = tf.tile([loop_var['gold_seq'][0]], [tf.shape(modified['actions'][1:])[0]])
                score_mask = tf.cast(tf.not_equal(modified['actions'][1:, -1], tiled_gold_action), tf.float32)
                adjusted_score = modified['score'][1:] + tf.multiply(score_mask, self.params['margin'])
                # iter_loss = tf.maximum(0.0, (tf.reduce_max(adjusted_score) - gold_score))
                iter_loss = tf.maximum(0.0, tf.reduce_max(adjusted_score) - gold_score)
                # iter_loss = tf.Print(iter_loss, [gold_score, score_mask, max_incorrect_score], summarize=200)
                modified['loss'] = tf.concat([loop_var['loss'], [iter_loss]], axis=-1)
                
            modified['buffer_char'] = loop_var['buffer_char'][1:]
            modified['gold_seq'] = loop_var['gold_seq'][1:]
            modified['run_mode'] = loop_var['run_mode']
            return tuple([modified[key] for key in var_list])
        
        def loop_cond(*args):
            buffer_bwd_len = args[var_list.index('buffer_bwd_len')]
            score = args[var_list.index('score')]
            len_cond = tf.not_equal(buffer_bwd_len[0], 0)
            score_cond = tf.greater_equal(score[0], score[-1])
            # score_cond = tf.Print(score_cond, [score[0], score[-1]], 'score', summarize=100)
            train_cond = args[var_list.index('run_mode')]
            return tf.cond(train_cond, lambda: tf.logical_and(len_cond, score_cond), lambda: len_cond)
        
        if run_mode == 'TRAIN':
            gold_seq = self.gold_sequence
        else:
            gold_seq = tf.concat([tf.constant([2], dtype=tf.int32), tf.tile([3], [tf.shape(self.buffer_char)[0] - 1])], axis=-1)
        loop_args = tuple([tf.constant([0.0, 0.0], dtype=tf.float32), tf.constant([], dtype=tf.string), 
                           self.buffer_char, gold_seq, tf.constant([], dtype=tf.float32)] 
                              + [self.x[key] for key in var_list[5:-1]] 
                              + [tf.constant(run_mode == 'TRAIN', tf.bool)])
        
        shape_list = tuple([tensor_shape[key] for key in var_list])
        beam_result = tf.while_loop(loop_cond, loop_body, loop_args, shape_invariants=shape_list)
        output_dict = dict((var_list[i], beam_result[i]) for i in range(len(var_list)))
        return output_dict
    
    def define_output_dict(self):
        trained_output_dict = self.run_beam('TRAIN')
        predicted_output_dict = self.run_beam('PREDICT')
        
        self.trained_out_actions = trained_output_dict['actions']
        self.trained_out_score = trained_output_dict['score']
        trained_loss_list = trained_output_dict['loss']
        
        self.predicted_out_actions = predicted_output_dict['actions']
        self.predicted_out_score = predicted_output_dict['score']
        predicted_loss_list = predicted_output_dict['loss']
        
        self.trained_beam_loss = tf.reduce_mean(trained_loss_list) + self.reg_loss
        self.predicted_beam_loss = tf.reduce_mean(predicted_loss_list) + self.reg_loss
