from preprocess import read_sentence_from_file
from embedding import Embeddings
import sys
from collections import Counter


def prepare_bigram(file_path, emb, output_path):
    sentences = read_sentence_from_file(file_path)
    all_pairs = list()

    for sentence in sentences:
        appended_sentence = ([emb.start_keyword] * 2) + \
                            [c for word in sentence for c in word] + \
                            ([emb.end_keyword] * 2)
        emb_sentence = []
        for token in appended_sentence:
            if emb.is_keyword_in_emb(token):
                emb_sentence.append(token)
            else:
                emb_sentence.append(emb.unk_keyword)

        for i in range(0, len(emb_sentence) - 1):
            all_pairs.append((emb_sentence[i], emb_sentence[i+1]))

    counter = Counter(all_pairs)
    selected_bigrams = list()
    for bigram in counter:
        if counter[bigram] > 1:
            selected_bigrams.append(bigram)

    sorted_bigrams = sorted(selected_bigrams, key=lambda x: x[0])
    with open(output_path, 'w') as output_file:
        for bigram in sorted_bigrams:
            output_file.write(bigram[0] + '\t' + bigram[1] + '\n')


if __name__ == '__main__':
    train_file_path = sys.argv[1]
    char_emb = Embeddings('embeddings/zh/polyglot-zh_char.dict',
                             pre_trained_file_path='embeddings/zh/polyglot-zh_char.emb.npy',
                             unk_keyword='<UNK>', pad_keyword='<PAD>', start_keyword='<S>', end_keyword='</S>')
    prepare_bigram(train_file_path, char_emb, 'embeddings/zh/zh_bigram.dict')


