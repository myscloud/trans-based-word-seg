import tensorflow as tf
from word_seg_model import WordSegModel


def simple_model_fn(mode, input_reader, model_path, network_params,
                    epoch_no=None, iteration_no=None, pretrained=None, eval_input_fn=None):
    if mode == 'TRAIN':
        pass
    elif mode == 'PREDICT':
        epoch_no = 1
    else:
        raise Exception('Invalid mode: ' + mode)

    iteration_count = 0
    epoch_count = 0

    with tf.Session() as sess:
        model = WordSegModel(network_params)

        # init model
        sess.run(model.init_model())
        emb_feed_dict = {model.emb_pl['word']: pretrained['word'], model.emb_pl['char']: pretrained['char']}
        sess.run(model.init_embeddings(), feed_dict=emb_feed_dict)

        while True:
            epoch_end = False
            while not epoch_end:
                data_dict, epoch_end = input_reader.get_next()
                feed_dict = get_feed_dict(model, data_dict, labels=data_dict['action_label'])
                loss = sess.run([model.simple_loss], feed_dict=feed_dict)
                iteration_count += 1

            epoch_count += 1


def beam_model_fn(mode, input_reader, network_params, model_path,
                  beam_size, decode_size, summaries_path=None,
                  epoch_no=None, iteration_no=None, pretrained=None, eval_input_fn=None, eval_every_epoch=1):
    """
    :param mode: TRAIN or EVALUATE or PREDICT
    :param input_reader: An instance of input_reader.BeamInputReader
    :param network_params:
    :param model_path: path for a model to be saved or restored
    :param beam_size:
    :param decode_size:
    :param epoch_no: int or None
    :param iteration_no: int or None (if both epoch_no and iteration_no are None, it will run infinitely)
    :param pretrained: pre-trained embeddings
    :param eval_input_fn: An instance of input_reader.BeamInputReader for evaluation (in training process)
    :param eval_every_epoch: In training mode, specify that eval_input_fn should be run every x epoch
    :return: in TRAIN mode return nothing,
            in PREDICT mode return decoded sentences (first decode_size scored sequences for each sentence)
    """
    end_epoch = False
    end_process = False
    iteration_count = 0
    epoch_count = 0
    decoded_results = list()

    if mode == 'TRAIN':
        if model_path is None or summaries_path is None:
            raise Exception('TRAIN mode: use must define model_save_path, summaries_path')
    elif mode == 'PREDICT':
        pass
    else:
        raise Exception('Mode can be only TRAIN or PREDICT')

    with tf.Session() as sess:
        model = WordSegModel(network_params)
        sess.run(model.init_model())

        saver = tf.train.Saver()
        model_save_path = model_path + '/model.ckpt'
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # ----------------------
            emb_feed_dict = {model.emb_pl['word']: pretrained['word'], model.emb_pl['char']: pretrained['char']}
            sess.run(model.init_embeddings(), feed_dict=emb_feed_dict)
            sess.run(model.reset_accumulated_gradients())
            # ----------------------

        writer = tf.summary.FileWriter(summaries_path, sess.graph)
        initial_index = sess.run(model.global_step)
        print(initial_index)

        while not end_process:
            while not end_epoch:
                # ----------------------
                beam, end_epoch, states = input_reader.get_sentence()
                sentence_len = len(beam[0]['buffer_char'])
                loss_sum = 0.0

                for _ in range(sentence_len):
                    # training
                    batch_beam, label, mask = input_reader.make_batch(beam)
                    feed_dict = get_feed_dict(model, batch_beam, label, mask)

                    operating_ops = [model.accumulate_gradient(model.max_margin_loss_fn),
                                     model.max_margin_loss, model.score_sep, model.score_app]
                    _, loss, score_sep, score_app = sess.run(operating_ops, feed_dict=feed_dict)
                    loss_sum += loss

                    # beam search
                    new_beam = list()
                    for i, state in enumerate(beam):
                        last_score = state['score']
                        new_beam.append({'state': state, 'next_action': 'SEP', 'score': last_score + score_sep[i]})
                        new_beam.append({'state': state, 'next_action': 'APP', 'score': last_score + score_app[i]})

                    sorted_beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)
                    pruned_beam = sorted_beam[0: beam_size]
                    beam = input_reader.get_next_beam(pruned_beam)

                # summary
                train_summary = dict()
                train_summary['loss'] = loss_sum / sentence_len
                write_summary(writer, train_summary, iteration_count)

                sess.run(model.average_gradients(), feed_dict={model.no_steps: float(sentence_len)})
                sess.run(model.apply_gradients())
                sess.run(model.reset_accumulated_gradients())

                # ----------------------

                if mode == 'PREDICT':
                    # put results to the list
                    results = [state['stack_word'] for state in beam[0:decode_size]]
                    decoded_results.append(results)
                elif mode == 'TRAIN':
                    iteration_count += 1
                    if iteration_no % eval_every_epoch == 0:
                        pass
                    if iteration_no is not None and iteration_count >= iteration_no:
                        end_process = True
                        break

            if mode == 'TRAIN':
                saver.save(sess, model_save_path, global_step=(initial_index + epoch_count))
                epoch_count += 1
                if epoch_no is not None and epoch_count >= epoch_no:
                    end_process = True
            else:
                end_process = True

    if mode == 'PREDICT':
        return decoded_results


def get_feed_dict(model, input_dict, labels=None, mask=None):
    feed_dict = dict()
    for input_tag in model.input:
        feed_dict[model.input[input_tag]] = input_dict[input_tag]
    if labels is not None:
        feed_dict[model.labels] = labels
    if mask is not None:
        feed_dict[model.mask] = mask
    return feed_dict


def write_summary(writer, value_pair, global_step):
    summary = tf.Summary()
    for key in value_pair:
        summary.value.add(tag=key, simple_value=value_pair[key])

    writer.add_summary(summary, global_step)
