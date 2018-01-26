import pickle
import numpy as np


class SimpleInputReader:
    def __init__(self, file_path, embeddings, batch_size=32, get_first_only=False, shuffle=False, pad_batch=False):
        self.embeddings = embeddings
        self.data, self.raw_data = get_input_label(file_path, embeddings)

        self.batch_size = batch_size
        self.data_size = len(self.data)
        self.get_first_only = get_first_only
        self.is_shuffle = shuffle
        self.epoch_end = False
        self.iterator = 0
        self.pad_batch = pad_batch

        if self.get_first_only:
            self.batch_data = [sent[0] for sent in self.data]
        else:
            self.batch_data = [state for sent in self.data for state in sent]

        self.data_size = len(self.batch_data)
        self.batch_data = self.pad_sequence()
        self.original_batch_data = self.batch_data.copy()
        self.current_gold_data = None

        if shuffle:
            self._shuffle_data()

    def pad_sequence(self):
        pad_list = [('buffer', self.embeddings['char'].get_pad_index()),
                    ('stack', self.embeddings['word'].get_pad_index()),
                    ('actions', self.embeddings['action'].get_pad_index()),
                    ('bigram', self.embeddings['bigram'].get_pad_index())]

        element_name = [key for key in self.batch_data[0]]
        element_list = dict([(key, []) for key in element_name])
        for record in self.batch_data:
            for element in element_name:
                element_list[element].append(record[element])

        if self.pad_batch:
            for (element, pad_key) in pad_list:
                len_list = [len(record) for record in element_list[element]]
                max_len = max(len_list)

                for i, record in enumerate(element_list[element]):
                    element_list[element][i] = record + [pad_key] * (max_len - len(record))

        element_dict = dict()
        for element in element_list:
            if type(element) is list:
                element_dict[element] = np.vstack(tuple(element_list[element]))
            else:
                element_dict[element] = np.array(element_list[element])

        return element_dict

    def start_new_epoch(self):
        if self.is_shuffle:
            self._shuffle_data()

        self.epoch_end = False
        self.iterator = 0

    def is_epoch_end(self):
        return self.epoch_end

    def _shuffle_data(self):
        if not self.is_shuffle:
            return self.batch_data

        shuffled_index = np.arange(self.data_size)
        np.random.shuffle(shuffled_index)
        for element in self.batch_data:
            self.batch_data[element] = self.original_batch_data[element][shuffled_index]

    def get_next(self):
        next_item = dict()

        if self.iterator + self.batch_size >= self.data_size:
            for element in self.batch_data:
                next_item[element] = self.batch_data[element][self.iterator:self.data_size]
            next_iterator = 0

            if self.pad_batch:
                next_iterator = self.batch_size - len(next_item)
                for element in self.batch_data:
                    next_item[element] = np.concatenate((next_item[element],
                                                         self.batch_data[element][0:next_iterator]), axis=0)

            self.epoch_end = True

        else:
            next_iterator = self.iterator + self.batch_size
            for element in self.batch_data:
                next_item[element] = self.batch_data[element][self.iterator:next_iterator]

        returned_keys = ['stack', 'buffer', 'bigram', 'actions', 'buffer_len', 'buffer_fwd_len', 'buffer_bwd_len', 'stack_len', 'actions_len']
        selected_next_item = dict([(key, next_item[key]) for key in returned_keys])
        self.current_gold_data = next_item['action_label']
        self.iterator = next_iterator
        return selected_next_item, next_item['action_label']

    def get_gold_data(self):
        if self.current_gold_data is None:
            raise Exception('There is problems retrieving gold_data.')
        return self.current_gold_data.tolist()


class BeamInputReader:
    def __init__(self, file_path, embeddings, shuffle=False):
        self.embeddings = embeddings
        self.data_label, self.raw_data = get_input_label(file_path, embeddings)
        self.batch_data = self.make_batch(self.data_label)
        self.is_shuffle = shuffle
        self.data_size = len(self.data_label)

        self.shuffled_index = np.arange(len(self.data_label))
        self.shuffled_data = self.batch_data
        if shuffle:
            self._shuffle()
        self.current_sentence = 0
        self.current_state_count = 0
        self.batch_end = False
        self.batch_size = 1
        self.sequence_element = ['stack', 'actions', 'buffer', 'bigram', 'buffer_len',
                                 'buffer_fwd_len', 'buffer_bwd_len', 'stack_len', 'actions_len']

    def start_new_epoch(self):
        if self.is_shuffle:
            self._shuffle()

        self.batch_end = False
        self.current_sentence = -1
        self.current_state_count = 0

    def is_epoch_end(self):
        return self.batch_end

    def _shuffle(self):
        if not self.is_shuffle:
            return self.data_label

        self.shuffled_index = np.arange(len(self.data_label))
        np.random.shuffle(self.shuffled_index)
        self.shuffled_data = list()
        for idx in self.shuffled_index:
            self.shuffled_data.append(self.batch_data[idx])

    def get_next(self):
        if self.current_sentence + 1 == len(self.shuffled_data) - 1:
            self.batch_end = True

        self.current_sentence += 1
        self.current_state_count = 0
        next_sentence = self.shuffled_data[self.current_sentence]
        input_data = dict([(key, next_sentence[key] * 2) for key in next_sentence])
        input_data['buffer_char'] = next_sentence['buffer_char']
        
        # get gold sequence
        action_labels = list()
        index = self.shuffled_index[self.current_sentence]
        for word_info in self.data_label[index]:
            action_labels.append(word_info['action_label'])
        
        return input_data, action_labels

    def get_gold_data(self):
        index = self.shuffled_index[self.current_sentence]
        return [self.raw_data[index]]

    def get_next_beam(self, beam, sorted_index):
        """
        :param beam: a list of dict {'state': word_seg_state, 'next_action': next action to perform}
        :return: a list of new beam consisting states after performing each next_action
        """
        # fill next gold action to the gold sequence
        gold_state = self.shuffled_data[self.current_sentence][self.current_state_count]
        gold_action = self.embeddings['action'][gold_state['action_label']]
        beam[0]['next_action'] = gold_action
        next_gold_action = gold_state['action_label']
        
        # iterate
        next_beam = list()
        next_char = self.shuffled_data[self.current_sentence][0]['buffer_char'][self.current_state_count]
        
        next_stack = list()
        next_stack_len = list()
        new_index_list = list()
        
        for i, element in enumerate(beam):
            state = element['state']
            next_action = element['next_action']
            next_state = state.copy()

            if next_action == 'APP' and len(next_state['stack_word']) <= 0:
                pass
            else:
                next_state['stack_word'] = state['stack_word'].copy()
                next_state['stack'] = state['stack'].copy()

                if next_action == 'SEP':
                    next_state['stack_word'].append(next_char)
                    last_word = next_state['stack_word'][-1]
                    next_state['stack'].append(self.embeddings['word'][last_word])
                else:
                    next_state['stack_word'][-1] += next_char
                    last_word = next_state['stack_word'][-1]
                    next_state['stack'][-1] = self.embeddings['word'][last_word]

                next_state['actions'] = state['actions'].copy() + [self.embeddings['action'][next_action]]
                next_state['buffer_fwd_len'] += 1
                next_state['buffer_bwd_len'] -= 1
                next_state['stack_len'] = len(next_state['stack'])
                next_state['actions_len'] += 1

                next_beam.append(next_state)
                
                next_stack.append(next_state['stack'][-1])
                next_stack_len.append(next_state['stack_len'])
                if i > 0:
                    new_index_list.append(sorted_index[i-1])

        self.current_state_count += 1
        return next_beam, (next_stack, next_stack_len, next_gold_action, new_index_list)

    def make_batch(self, data_label):
        batch_data = list()
        batch_label = ['stack', 'buffer', 'bigram', 'actions', 'buffer_len', 'buffer_fwd_len', 'buffer_bwd_len', 'stack_len', 'actions_len']
        for sentence in data_label:
            batch_info = dict()
            for key in sentence[0]:
                if key in batch_label:
                    batch_info[key] = [sentence[0][key]]
                else:
                    batch_info[key] = sentence[0][key]
            batch_data.append(batch_info)
        
        return batch_data
    
    def get_stack_pad_key(self):
        return self.embeddings['word'].get_pad_index()
    
    def get_labels(self):
        gold_state = self.shuffled_data[self.current_sentence][self.current_state_count-1]
        label = [gold_state['action_label'] - 2]
        mask = [0.0, 0.0]
        mask[label[0]] = 1.0  # gold action
        return label, mask


def get_input_label(file_path, embeddings):
    with open(file_path, 'rb') as data_file:
        all_sentences = pickle.load(data_file)

    # pad symbol
    char_start, char_end = embeddings['char'].start_keyword, embeddings['char'].end_keyword
    word_start, word_end = embeddings['word'].start_keyword, embeddings['word'].end_keyword
    act_pad = embeddings['action'].start_keyword

    if char_start is None or char_end is None or word_start is None or word_end is None or act_pad is None:
        raise Exception('One of the char/word/action start or end keyword is not defined in embeddings.')

    dataset_label = list()
    for sentence in all_sentences:
        current_sentence = list()
        actions = [embeddings['action'][act_pad]] * 2  # 'actions' means history of actions executed
        stack = [embeddings['word'][word_start]] * 2
        stack_word = []

        sentence_char = ([char_start] * 3) + [c for word in sentence for c in word] + ([char_end] * 2)
        buffer = [embeddings['char'][c] for c in sentence_char][1:]
        bigram = list()
        for i in range(len(sentence_char) - 1):
            bigram.append(embeddings['bigram'][(sentence_char[i], sentence_char[i+1])])

        tmp_buffer = sentence.copy()
        char_counter = 1
        all_char_len = len(sentence_char) - 5
        next_action = 'SEP'

        while len(tmp_buffer) > 0:
            action_label = embeddings['action'][next_action]
            current_sentence.append({
                'stack': stack.copy(),
                'stack_word': stack_word.copy(),

                'buffer': buffer.copy(),
                'buffer_char': sentence_char[3:-2],
                'bigram': bigram.copy(),

                'actions': actions.copy(),

                'buffer_len': all_char_len,
                'buffer_fwd_len': char_counter,
                'buffer_bwd_len': all_char_len - char_counter + 1,

                'stack_len': len(stack) - 1,
                'actions_len': len(actions) - 1,

                'action_label': action_label,
            })

            # for the next state
            char_counter += 1
            actions.append(action_label)
            if next_action == 'SEP':
                stack_word.append(tmp_buffer[0][0])
                stack.append(embeddings['word'][stack_word[-1]])
            elif next_action == 'APP':
                stack_word[-1] += tmp_buffer[0][0]
                stack[-1] = embeddings['word'][stack_word[-1]]

            tmp_buffer[0] = tmp_buffer[0][1:]
            if len(tmp_buffer[0]) == 0:
                del tmp_buffer[0]
                next_action = 'SEP'
            else:
                next_action = 'APP'

        dataset_label.append(current_sentence)

    return dataset_label, all_sentences
