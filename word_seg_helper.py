import time

def simple_model_fn(sess, model, mode, inputs, labels, params, iteration):
    feed_dict = _get_feed_dict(model, input_dict=inputs, labels=labels)
    if mode == 'TRAIN':
        model.drop_embeddings(mode)
        _, loss, score_sep, score_app = sess.run([model.simple_optimizer, model.simple_loss, model.score_sep, model.score_app], feed_dict=feed_dict)
    else:
        loss, score_sep, score_app = sess.run([model.simple_loss, model.score_sep, model.score_app], feed_dict=feed_dict)
        
    answer_list = list()
    for sep, app in zip(score_sep, score_app):
        answer_index = 2 if sep > app else 3
        answer_list.append(answer_index)
    return loss, {'result': answer_list}

def beam_model_fn(sess, model, mode, inputs, labels, params, iteration):
    if mode == 'TRAIN':
        feed_dict = _get_feed_dict(model, input_dict=inputs, gold_sequence=labels)
        train_ops = [model.grad_assign_ops, model.trained_beam_loss, model.trained_out_actions, model.trained_out_score]
        _, loss, action_list, score = sess.run(train_ops, feed_dict=feed_dict)
        if score[0] < score[-1] or ((len(action_list[0]) - 2 == len(inputs['buffer_char'])) and score[0] < score[1]):
            sess.run(model.accu_grad_assign_ops)
        if iteration % params['beam_batch_size'] == 0:
            sess.run(model.beam_optimize)
            sess.run(model.reset_accu_grad_ops)
    elif mode == 'EVAL':
        feed_dict = _get_feed_dict(model, input_dict=inputs, gold_sequence=labels)
        loss, action_list = sess.run([model.predicted_beam_loss, model.predicted_out_actions], feed_dict=feed_dict)
    else:
        feed_dict = _get_feed_dict(model, input_dict=inputs)
        action_list = sess.run(model.predicted_out_actions, feed_dict=feed_dict)
        loss = 0.0

    result_list = list()
    first_result = list()
    if mode == 'EVAL' or  mode == 'PREDICT':
        buffer_char = inputs['buffer_char']
        decode_size = min(params['decode_size']+1, len(action_list))
        for sequence in action_list[1:decode_size]:
            possible_segment = list()
            for i, action in enumerate(sequence[2:]):
                if action == 2:
                    possible_segment.append(buffer_char[i])  # SEP
                else:
                    possible_segment[-1] += buffer_char[i]  # APP
            result_list.append(possible_segment)
        first_result = [result_list[0]]
    
    return loss, {'result': first_result, 'all_segment': result_list}

def summary_fn(loss, eval_dict):
    highlight_value = ['f1_score', 'accuracy']
    summary_dict = dict()
    summary_dict['loss'] = loss

    for key in eval_dict:
        if isinstance(eval_dict[key], int) or isinstance(eval_dict[key], float):
            summary_dict[key] = eval_dict[key]

    display_text = 'loss: %.5f' % loss
    for value in highlight_value:
        if value in eval_dict:
            display_text += ', %s: %.5f' % (value, eval_dict[value])

    return summary_dict, display_text


def pretrain_fn(sess, model, pretrain_dict):
    pretrained = pretrain_dict['pretrained']
    emb_feed_dict = {model.emb_pl['word']: pretrained['word'], model.emb_pl['char']: pretrained['char']}
    sess.run(model.init_embeddings(), feed_dict=emb_feed_dict)
    if 'beam_size' in pretrain_dict:
        sess.run(model.assign_beam_size(pretrain_dict['beam_size']))

def _get_feed_dict(model, input_dict, labels=None, gold_sequence=None):
    feed_dict = dict()
    for input_tag in model.x:
        feed_dict[model.x[input_tag]] = input_dict[input_tag]
    if 'buffer_char' in input_dict:
        feed_dict[model.buffer_char] = input_dict['buffer_char']
    if labels is not None:
        feed_dict[model.labels] = labels
    if gold_sequence is not None:
        feed_dict[model.gold_sequence] = gold_sequence
    return feed_dict
