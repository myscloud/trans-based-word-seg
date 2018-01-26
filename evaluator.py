
def evaluate_fn(batch_results, gold_results):
    if len(batch_results) == 0:
        return {}, []
    if len(batch_results) != len(gold_results):
        raise Exception('No. of records in batch_results and gold_results are not equal.')
    elif len(batch_results) == 0:
        raise Exception('There is no data in batch_results or gold_results to be evaluated.')

    if isinstance(batch_results[0], list):
        return evaluate_sentence(batch_results, gold_results)
    elif isinstance(batch_results[0], int):
        return evaluate_actions(batch_results, gold_results)
    else:
        raise Exception('Type of predicted results are not matched with the allowed type (int/list).')


def evaluate_actions(batch_results, gold_results):
    correct_count = 0
    correct_list = list()
    for predicted, gold in zip(batch_results, gold_results):
        if predicted == gold:
            correct_count += 1
        correct_list.append(predicted == gold)

    accuracy = (correct_count / len(batch_results)) * 100
    return {'correct_count': correct_count, 'all_count': len(batch_results), 'accuracy': accuracy}, [{'correct_list': correct_list}]


def evaluate_sentence(batch_results, gold_results):
    sent_eval_list = list()

    for predicted_sent, gold_sent in zip(batch_results, gold_results):
        sent_eval = evaluate_each_sentence(predicted_sent, gold_sent)
        sent_eval_list.append(sent_eval)

    batch_eval_list = {key: [] for key in sent_eval_list[0]}
    for sent_eval in sent_eval_list:
        for key in batch_eval_list:
            batch_eval_list[key].append(sent_eval[key])

    batch_eval_dict = dict()
    for key in ['true_positive', 'false_positive', 'false_negative']:
        batch_eval_dict[key] = sum(batch_eval_list[key])
    batch_eval_dict['precision'] = (batch_eval_dict['true_positive'] / (batch_eval_dict['true_positive'] + batch_eval_dict['false_positive'])) * 100
    batch_eval_dict['recall'] = (batch_eval_dict['true_positive'] / (batch_eval_dict['true_positive'] + batch_eval_dict['false_negative'])) * 100
    if (batch_eval_dict['precision'] + batch_eval_dict['recall']) > 0:
        batch_eval_dict['f1_score'] = (2 * batch_eval_dict['precision'] * batch_eval_dict['recall']) / (batch_eval_dict['precision'] + batch_eval_dict['recall'])
    else:
        batch_eval_dict['f1_score'] = 0.0

    batch_eval_dict['no_eval_items'] = len(batch_results)
    return batch_eval_dict, sent_eval_list


def evaluate_each_sentence(predicted_sent, gold_sent):
    def generate_start_end_map(data_list):
        n_len = sum([len(word) for word in data_list])
        data_map = [{'end': None, 'word_idx': None} for _ in range(n_len)]

        last_idx = 0
        for word_idx, word in enumerate(data_list):
            start_idx = last_idx
            end_idx = start_idx + len(word)

            data_map[start_idx]['end'] = end_idx
            data_map[start_idx]['word_idx'] = word_idx

            last_idx = end_idx

        return data_map

    predicted_map = generate_start_end_map(predicted_sent)
    gold_map = generate_start_end_map(gold_sent)

    # count no. of correctly segmented words
    correct_count = 0
    for start_idx in range(len(gold_map)):
        end_idx = gold_map[start_idx]['end']
        if end_idx is not None and predicted_map[start_idx]['end'] == end_idx:
            correct_count += 1

    eval_dict = dict()
    eval_dict['correct'] = correct_count
    eval_dict['gold_len'] = len(gold_sent)
    eval_dict['predicted_len'] = len(predicted_sent)
    eval_dict['true_positive'] = correct_count
    eval_dict['false_positive'] = len(predicted_sent) - correct_count
    eval_dict['false_negative'] = len(gold_sent) - correct_count
    eval_dict['precision'] = precision = (correct_count / len(predicted_sent)) * 100
    eval_dict['recall'] = recall = (correct_count / len(gold_sent)) * 100
    eval_dict['f1_score'] = ((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

    return eval_dict
