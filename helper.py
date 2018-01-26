import datetime
import os
import sys
from shutil import copyfile

options = {
    # network hyperparameters
    'network': {
        'learning_rate': 10e-7,
        'simple_dropout_keep_rate': 0.2,
        'emb_dropout_keep_rate': 0.5,
        'beam_dropout_keep_rate': 0.5,
        'margin': 0.2,
        'reg_const': 10e-8,
        
        'simple_batch_size': 256,
        'beam_batch_size': 1,

        # vocab size (for embeddings)
        'word_vocab_size': 100004,
        'char_vocab_size': 100004,
        'bigram_vocab_size': 65458,
        'action_size': 2,

        # embeddings
        'word_emb_size': 64,
        'char_emb_size': 64,
        'bigram_emb_size': 64,
        'action_emb_size': 20,

        # dimensions
        'rep_buffer': 50,
        'rep_stack': 50,
        'rep_actions': 20,
        'lstm_word': 50,
        'lstm_fwd_char': 50,
        'lstm_bwd_char': 50,
        'lstm_actions': 20,
        'hidden_sep': 100,
        'hidden_app': 80,

        # beam search
        'beam_size': 1,
        'decode_size': 3,
    },
    'experiment_name': '48-ud-drop',
    'run_description': 'predict-165',
    'embeddings': {
      'char': dict(index_file_path='embeddings/zh/polyglot-zh_char.dict',
              pre_trained_file_path='embeddings/zh/polyglot-zh_char.emb.npy',
              unk_keyword='<UNK>', pad_keyword='<PAD>', start_keyword='<S>', end_keyword='</S>'),
        'word': dict(index_file_path='embeddings/zh/polyglot-zh.dict', 
                     pre_trained_file_path='embeddings/zh/polyglot-zh.emb.npy',
                     unk_keyword='<UNK>', pad_keyword='<PAD>', start_keyword='<S>', end_keyword='</S>'),
        'action': dict(index_file_path='embeddings/actions.dict', 
                       skip_zero=True, start_keyword='<START>'),
        'bigram': dict(index_file_path='embeddings/zh/zh_bigram.dict', 
                       delimiter='\t', unk_keyword=('<UNK>', '<UNK>'), pad_keyword=('<PAD>', '<PAD>'))
    },
    # 'input_path': {
    #     'train': 'data/sinica_train.pkl',
    #     'train_eval': 'data/sinica_eval.pkl',
    #     'dev': 'data/sinica_dev.pkl',
    #     'test': 'data/sinica_test.pkl',
    #     'dummy': 'data/sinica_mini_eval.pkl'
    # },
    'input_path': {
        'train': 'data/ud-train.pkl',
        'train_eval': 'data/ud-eval.pkl',
        'dev': 'data/ud-dev.pkl',
        'test': 'data/ud-test.pkl',
        'dummy': 'data/sinica_mini_eval.pkl'
    },
}


class Logger(object):
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        if '*' not in message:
            self.log_file.write(message)

    def flush(self):
        pass

def create_experiment_folder(exp_name, run_name):
    new_exp_path = 'experiments/' + exp_name + '/'
    if not os.path.exists(new_exp_path):
        os.makedirs(new_exp_path)
        os.makedirs(new_exp_path + 'model')
        os.makedirs(new_exp_path + 'summaries')
        os.makedirs(new_exp_path + 'logs')
        os.makedirs(new_exp_path + 'results')
        os.makedirs(new_exp_path + 'code')
    
    time_suffix = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    new_code_path = new_exp_path + 'code/' + time_suffix + '/'
    os.makedirs(new_code_path)

    files = [f for f in os.listdir(os.getcwd()) if os.path.isfile(f)]
    for name in files:
        if name[-3:] == '.py':
            file_path = os.path.join(os.getcwd(), name)
            new_file_path = new_code_path + name
            if not os.path.isfile(new_file_path):
                copyfile(file_path, new_file_path)

    print('Results and models will be stored in', new_exp_path)
    return new_exp_path


def init_experiment():
    run_name = options['run_description']
    exp_path = create_experiment_folder(options['experiment_name'], run_name)
    options['model_path'] = exp_path + 'model/'
    options['result_path'] = exp_path + 'results/'
    options['summary_path'] = exp_path + 'summaries/'
    options['network']['experiment_name'] = options['run_description']  # don't worry :)

    log_file_path = exp_path + 'logs/' + run_name + '.log'
    sys.stdout = Logger(log_file_path)
    print('\n============================================')
    print('Running at', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    print('============================================')

    return options
