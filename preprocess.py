import argparse
import pickle


parser = argparse.ArgumentParser(description='Preprocess file in CoNLL format for word segmentation task.')
parser.add_argument('-i', action='store', dest='input_path', type=str)
parser.add_argument('-o', action='store', dest='output_path', type=str)


def read_sentence_from_file(file_path):
    all_sentence = list()
    current_sentence = list()

    with open(file_path) as data_file:
        for line in data_file:
            if len(line) < 2:
                all_sentence.append(current_sentence)
                current_sentence = list()
            elif line[0] != '#':
                tokens = line.split('\t')
                current_sentence.append(tokens[1])

        if len(current_sentence) > 0:
            all_sentence.append(current_sentence)

    return all_sentence


if __name__ == '__main__':
    arg_dict = vars(parser.parse_args())

    all_sent = read_sentence_from_file(arg_dict['input_path'])
    with open(arg_dict['output_path'], 'wb') as out_file:
        pickle.dump(all_sent, out_file)
