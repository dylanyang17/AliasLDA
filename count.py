from lda_mh_alias import Lda_MH_Alias


def count_words_per_doc(dataset, percentage):
    model = Lda_MH_Alias()
    model.load_data_formal(filename='data/docword.%s.txt/docword.%s.txt' % (dataset, dataset), percentage=percentage)
    filepath = 'data/docword.%s.txt/word_num_per_doc.%s%d%%.txt' % (dataset, dataset, percentage)
    with open(filepath, 'w') as f:
        for num in model.word_num_per_doc:
            print(num, file=f)


if __name__ == '__main__':
    count_words_per_doc('nips', 50)
