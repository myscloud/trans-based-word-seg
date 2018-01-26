from embedding import Embeddings
from input_reader import SimpleInputReader, BeamInputReader
from helper import init_experiment
from evaluator import evaluate_fn
from custom_monitor import Monitor
import word_seg_helper as word_seg
from model import WordSegModel
from beam_model import BeamWordSegModel

import argparse


parser = argparse.ArgumentParser(description='Train word segmentation.')
parser.add_argument('-s', action='store_true', dest='is_simple')
parser.add_argument('-b', action='store_true', dest='is_beam')


if __name__ == '__main__':
    options = init_experiment()
    arg_dict = vars(parser.parse_args())

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
    
    if arg_dict['is_beam']:
        # Beam Training (max margin loss + beam)
        print('Running beam training...')
        beam_input_reader = BeamInputReader(options['input_path']['train'], emb, shuffle=True)
        beam_val_input_reader = BeamInputReader(options['input_path']['train_eval'], emb)

        model_fn = word_seg.beam_model_fn
        monitor = Monitor(model_fn=model_fn,
                          evaluate_fn=evaluate_fn,
                          summary_fn=word_seg.summary_fn,
                          pretrain_fn=word_seg.pretrain_fn,
                          model_instance=BeamWordSegModel,
                          model_path=options['model_path'],
                          summary_path=options['summary_path'],
                          params=options['network'])
        monitor.train(beam_input_reader, beam_val_input_reader, pretrain=pretrained, val_every_iter=250)
    else:
        # Simple Training (use cross entropy loss + greedy)
        print('Running simple training...')
        train_input_reader = SimpleInputReader(options['input_path']['train'], emb, shuffle=True, 
                                               pad_batch=True, batch_size=options['network']['simple_batch_size'])
        val_input_reader = SimpleInputReader(options['input_path']['train_eval'], emb, shuffle=False, 
                                             pad_batch=True, batch_size=options['network']['simple_batch_size'])
        model_fn = word_seg.simple_model_fn
        monitor = Monitor(model_fn=model_fn,
                          evaluate_fn=evaluate_fn,
                          summary_fn=word_seg.summary_fn,
                          pretrain_fn=word_seg.pretrain_fn,
                          model_instance=WordSegModel,
                          model_path=options['model_path'],
                          summary_path=options['summary_path'],
                          params=options['network'])

        monitor.train(train_input_reader, val_input_reader, pretrain=pretrained, val_every_iter=5000)  
