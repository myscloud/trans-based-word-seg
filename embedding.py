import numpy as np


class Embeddings:
    def __init__(self, index_file_path, skip_zero=False, delimiter=None, pre_trained_file_path=None,
                 unk_keyword=None, pad_keyword = None, start_keyword=None, end_keyword=None):
        self.index_to_token, self.token_to_index = Embeddings.read_index_file(index_file_path, delimiter, skip_zero)
        if pre_trained_file_path is not None:
            self.emb = np.load(pre_trained_file_path)
            if skip_zero:
                self.emb = np.concatenate((np.zeros((1, self.emb.shape[1])), self.emb))

        self.unk_keyword = self.validate_keyword('unk', unk_keyword)
        self.start_keyword = self.validate_keyword('start', start_keyword)
        self.end_keyword = self.validate_keyword('end', end_keyword)
        self.pad_keyword = self.validate_keyword('pad', pad_keyword) if not skip_zero else '-pad-'

    @staticmethod
    def read_index_file(index_file_path, delimiter, skip_zero):
        idx_tok_dict = dict()
        tok_idx_dict = dict()
        index = 0
        if skip_zero:
            idx_tok_dict[0] = '-pad-'
            tok_idx_dict['-pad-'] = 0
            index += 1

        with open(index_file_path) as index_file:
            for line in index_file:
                if delimiter is not None:
                    tokens = tuple(line.strip().split(delimiter))
                else:
                    tokens = line.strip()

                idx_tok_dict[index] = tokens
                tok_idx_dict[tokens] = index
                index += 1

        return idx_tok_dict, tok_idx_dict

    def validate_keyword(self, keyword_name, keyword):
        if keyword is None:
            return keyword

        if keyword not in self.token_to_index:
            raise Exception(keyword_name + ' keyword is not embedding: ' + str(keyword))

        return keyword

    def is_keyword_in_emb(self, keyword):
        return keyword in self.token_to_index

    def size(self):
        return len(self.token_to_index)

    # get special tokens' index
    def get_pad_index(self):
        if self.pad_keyword is None:
            raise Exception('There is no PAD keyword in the embeddings')
        return self.token_to_index[self.pad_keyword]

    def get_unk_index(self):
        if self.unk_keyword is None:
            raise Exception('There is no UNK keyword in the embeddings')
        return self.token_to_index[self.unk_keyword]

    def get_start_index(self):
        if self.start_keyword is None:
            raise Exception('There is no START keyword in the embeddings')
        return self.token_to_index[self.start_keyword]

    def get_end_index(self):
        if self.end_keyword is None:
            raise Exception('There is no END keyword in the embeddings')
        return self.token_to_index[self.end_keyword]
    
    def get_token_array(self):
        tokens = list()
        for i in range(len(self.index_to_token)):
            tokens.append(self.index_to_token[i])
        
        return tokens
    
    def get_index_array(self):
        return list(range(len(self.index_to_token)))

    # getitem: for using bracket to get index/token
    def __getitem__(self, item):
        if type(item) == int:
            if item in self.index_to_token:
                return self.index_to_token[item]
            else:
                raise Exception('Index out of range: ' + item)
        else:
            if item in self.token_to_index:
                return self.token_to_index[item]
            elif self.unk_keyword is not None:
                return self.token_to_index[self.unk_keyword]
            else:
                raise Exception('Given keyword is not in embeddings: ' + item)
