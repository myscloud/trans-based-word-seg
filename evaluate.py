from embedding import Embeddings
from input_reader import SimpleInputReader, BeamInputReader
from helper import init_experiment
from learner import simple_model_fn, beam_model_fn
from evaluator import evaluate_fn
from custom_monitor import Monitor
import word_seg_helper as word_seg
from beam_model import BeamWordSegModel

def write_overall_eval(eval_file_path, eval_dict):
    with open(eval_file_path, 'w') as eval_file:
        for eval_key in eval_dict:
            eval_file.write(eval_key + ':\t' + str(eval_dict[eval_key]) + '\n')

def write_detailed_csv(eval_file_path, eval_list):
    if len(eval_list) == 0:
        raise Exception('There is no evaluation in detailed evaluation list.')
    
    # get eval properties
    csv_head = list()
    for eval_key in eval_list[0]:
        csv_head.append(eval_key)
    
    with open(eval_file_path, 'w') as eval_file:
        # write header
        eval_file.write('no, ')
        for eval_key in csv_head:
            eval_file.write(eval_key + ', ')
        eval_file.write('\n')
        
        # write content
        for i, eval_dict in enumerate(eval_list):
            eval_file.write(str(i+1) + ', ')
            for eval_key in csv_head:
                eval_file.write(str(eval_dict[eval_key]) + ', ')
            eval_file.write('\n')

def write_word_seg_result(result_file_path, result):
    with open(result_file_path, 'w') as result_file:
        for i, sentence in enumerate(result):
            result_file.write('# Sentence ' + str(i+1) + '\n')
            for predicted in sentence['all_segment']:
                result_file.write(' '.join(predicted) + '\n')
            result_file.write('\n')
            
if __name__ == '__main__':
    options = init_experiment()
    
    emb = dict()
    embeddings_list = ['char', 'word', 'action', 'bigram']
    for emb_name in embeddings_list:
        emb[emb_name] = Embeddings(**options['embeddings'][emb_name])
    
    pretrained = {
        'pretrained': {'char': emb['char'].emb, 'word': emb['word'].emb},
        'beam_size': 10,
        'init': {'word_emb': (emb['word'].get_token_array(), emb['word'].get_index_array())}
    }

    options['network']['action_size'] = emb['action'].size()
    options['network']['bigram_vocab_size'] = emb['bigram'].size()
    
    input_reader = BeamInputReader(options['input_path']['dev'], emb, shuffle=False)
    monitor = Monitor(model_fn=word_seg.beam_model_fn,
                      evaluate_fn=evaluate_fn,
                      summary_fn=word_seg.summary_fn,
                      pretrain_fn=word_seg.pretrain_fn,
                      model_instance=BeamWordSegModel,
                      model_path=options['model_path'],
                      summary_path=options['summary_path'],
                      params=options['network'])
    
    eval_dict, detailed_eval_dict, result = monitor.evaluate(input_reader, pretrain=pretrained)
    overall_file_path = options['result_path'] + '/' + options['run_description'] + '-overall.txt'
    detailed_file_path = options['result_path'] + '/' + options['run_description'] + '-detailed.csv'
    result_file_path = options['result_path'] + '/' + options['run_description'] + '-word_seg_result.txt'
    write_overall_eval(overall_file_path, eval_dict)
    write_detailed_csv(detailed_file_path, detailed_eval_dict)
    write_word_seg_result(result_file_path, result)
    