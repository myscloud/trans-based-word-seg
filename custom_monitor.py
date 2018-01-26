import tensorflow as tf
import os
import time

config = tf.ConfigProto(allow_soft_placement=True)

class Monitor:
    def __init__(self,
                 model_fn,
                 evaluate_fn,
                 summary_fn,
                 pretrain_fn,
                 model_instance,
                 model_path,
                 summary_path,
                 params):
        if type(model_fn) is dict:
            self.model_fn = model_fn
        else:
            self.model_fn = {'default': model_fn}

        self.evaluate_fn = evaluate_fn
        self.summary_fn = summary_fn
        self.pretrain_fn = pretrain_fn
        self.model_instance = model_instance
        self.model_path = model_path
        self.summary_path = summary_path
        self.params = params

    def train(self, input_fn, val_input_fn, epoch_no=None, iteration_no=None, val_every_epoch=1, val_every_iter=None,
              pretrain=None, model_fn_name='default', val_model_fn_name='default'):
        self._run_model('TRAIN', input_fn, val_input_fn, epoch_no, iteration_no, val_every_epoch, val_every_iter, pretrain, model_fn_name, val_model_fn_name)

    def evaluate(self, input_fn, model_fn_name='default', pretrain=None):
        return self._run_model('EVAL', input_fn, epoch_no=1, model_fn_name=model_fn_name, pretrain=pretrain)

    def predict(self, input_fn, model_fn_name='default', pretrain=None):
        return self._run_model('PREDICT', input_fn, epoch_no=1, model_fn_name=model_fn_name, pretrain=pretrain)

    def _run_model(self,
                  mode,
                  input_fn,
                  val_input_fn=None,
                  epoch_no=None,
                  iteration_no=None,
                  val_every_epoch=None,
                  val_every_iter=None,
                  pretrain=None,
                  model_fn_name='default',
                  val_model_fn_name='default'
                  ):
        with tf.Session(config=config) as sess, tf.device('/cpu:0'):
            model = self.model_instance(self.params, pretrain['init'])
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            # create saver, restore session, and assign pre-trained value
            saver = tf.train.Saver(tf.trainable_variables() + [model.global_step])
            latest_checkpoint = tf.train.latest_checkpoint(self.model_path)
            model_save_path = self.model_path + '/model.ckpt'
            if latest_checkpoint is not None:
                saver.restore(sess, latest_checkpoint)
                print('Found the latest checkpoint at ', latest_checkpoint)
                print('Model is restored from ' + self.model_path + '.')
                sess.run(model.increase_global_step())
            else:
                print('Model will be saved at ' + self.model_path + '.')
                if pretrain is not None:
                    self.pretrain_fn(sess, model, pretrain)
                    print('Run pretrain function...')

            # create summary writers
            main_writer = val_writer = None
            if mode == 'TRAIN':
                main_writer = self._create_summary_writer(sess, 'train', self.params['experiment_name'])
                val_writer = self._create_summary_writer(sess, 'eval', self.params['experiment_name'])
            elif mode == 'EVAL':
                main_writer = self._create_summary_writer(sess, 'dev', self.params['experiment_name'])

            # select model fn
            model_fn = self.model_fn[model_fn_name]
            if mode == 'TRAIN':
                val_model_fn = self.model_fn[val_model_fn_name]
            else:
                val_model_fn = None

            # start running
            epoch_count = 0
            val_count = 0
            iteration_count = 0
            result_list = list()
            eval_dict = dict()
            all_eval_list = list()

            gold_val_result_list = list()
            gold_val_store = False

            initial_index = sess.run(model.global_step)
            end_running = False
            first_start_time = time.time()
            acc_train_time = 0.0
            acc_eval_time = 0.0
            run_count = 0

            while not end_running:
                loss_sum = 0.0
                result_list = list()
                eval_result_list = list()
                gold_result_list = list()
                loss_list = list()

                train_start_time = time.time()
                if input_fn.is_epoch_end() or run_count == 0:
                    print('Reset dataset')
                    input_fn.start_new_epoch()
                while not input_fn.is_epoch_end():
                    inputs, labels = input_fn.get_next()
                    loss, result_dict = model_fn(sess, model, mode, inputs, labels, self.params, iteration_count)
                    result = result_dict['result']
                    loss_list.append(loss)
                    gold_result_list.extend(input_fn.get_gold_data())
                    eval_result_list.extend(result)
                    if mode == 'EVAL' or mode == 'PREDICT':
                        result_list.append(result_dict)
                    run_count += 1
                    print('*Iteration ' + str(epoch_count) + ', ' + 
                          str(iteration_count%input_fn.data_size) + '/' + str(input_fn.data_size) + 
                          ' : loss = ' + str(loss), end='\r')

                    iteration_count += input_fn.batch_size

                    if mode == 'TRAIN':
                        loss_sum += (loss * input_fn.batch_size)
                        if iteration_no is not None and iteration_count >= iteration_no:
                            end_running = True
                            break
                        if val_every_iter is not None and (iteration_count >= (val_count * val_every_iter)):
                            break

                if input_fn.is_epoch_end():
                    epoch_count += 1

                # write summary for TRAIN and EVAL mode
                if mode == 'TRAIN' or mode == 'EVAL':
                    data_size = input_fn.data_size if val_every_iter is None else val_every_iter
                    average_loss = sum(loss_list) / len(loss_list)
                    eval_dict, all_eval_list = self.evaluate_fn(eval_result_list, gold_result_list)
                    for i, d in enumerate(all_eval_list):
                        d['loss'] = loss_list[i]
                    content, train_display_text = self.summary_fn(average_loss, eval_dict)
                    self._write_summary(main_writer, content, initial_index+val_count)
                    print('\n# Running ' + str(val_count) + ': ' + train_display_text)
                
                curr_iter_train_time = time.time() - train_start_time

                # save model
                if mode == 'TRAIN':
                    saver.save(sess, model_save_path, global_step=model.global_step)
                    sess.run(model.increase_global_step())

                # validation in training mode
                if mode == 'TRAIN' and (val_every_epoch is not None and epoch_count % val_every_epoch == 0) \
                or (val_every_iter is not None):
                    eval_start_time = time.time()
                    val_result_list = list()
                    val_input_fn.start_new_epoch()
                    loss_sum = 0.0
                    val_iter_count = 0
                    while not val_input_fn.is_epoch_end():
                        inputs, labels = val_input_fn.get_next()
                        loss, result_dict = val_model_fn(sess, model, 'EVAL', inputs, labels, self.params, val_iter_count)
                        result = result_dict['result']
                        loss_sum += loss
                        val_result_list.extend(result)
                        val_iter_count += val_input_fn.batch_size
                        print('*Evaluating ', val_iter_count ,' from ', val_input_fn.data_size, end='\r')
                        if not gold_val_store:
                            gold_val_result_list.extend(val_input_fn.get_gold_data())

                    gold_val_store = True
                    val_eval_dict, _ = self.evaluate_fn(val_result_list, gold_val_result_list)
                    average_loss = loss_sum / val_input_fn.data_size
                    content, val_display_text = self.summary_fn(average_loss, val_eval_dict)
                    print('\nEvaluation set: ' + val_display_text)
                    self._write_summary(val_writer, content, initial_index+val_count)
                    curr_iter_eval_time = time.time() - eval_start_time
                else:
                    curr_iter_eval_time = None

                # stopping condition
                val_count += 1
                if epoch_no is not None and epoch_count >= epoch_no:
                    end_running = True
                
                print('Epoch ', (iteration_count / input_fn.data_size))
                print('Mode:', mode, ', time used: ', curr_iter_train_time)
                if curr_iter_eval_time is not None:
                    print('Evaluation, time used: ', curr_iter_eval_time)
                print('Time used from the beginning of the process: ', time.time() - first_start_time)
                print('--------------------')
                

            if mode == 'EVAL':
                return eval_dict, all_eval_list, result_list
            elif mode == 'PREDICT':
                return result_list

    def _create_summary_writer(self, sess, run_mode, exp_name):
        summary_path = self.summary_path + '/' + run_mode + '-' + exp_name
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        writer = tf.summary.FileWriter(summary_path, sess.graph)
        return writer

    @staticmethod
    def _write_summary(writer, value_pair, global_step):
        summary = tf.Summary()
        for key in value_pair:
            summary.value.add(tag=key, simple_value=value_pair[key])

        writer.add_summary(summary, global_step)
