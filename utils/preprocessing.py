import numpy as np
import pandas as pd
import contractions
import string
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class WordEmbeddings():
    def __init__(self):
        pass

    def get_glove_vocab(self, twitter=False, root=''):
        filename = 'glove.6B.50d.txt' if not twitter else 'glove.twitter.27B.50d.txt'
        self.vocab = []

        with open(root+filename, encoding='utf-8') as file:
            for line in file:
                self.vocab.append(line.split()[0])

        return self.vocab

    def get_glove_embeddings(self, input_dim, vec_len, tokenizer, twitter=False, root=''):
        self.tokenizer = tokenizer
        self.input_dim = input_dim

        vec_len = vec_len if not twitter else 50
        if twitter and vec_len != 50:
            print('for twitter vec_len = 50')
        
        filename = fr'glove.6B.{vec_len}d.txt' if not twitter else 'glove.twitter.27B.50d.txt'
        self.embedding_matrix = np.zeros((input_dim, vec_len))
        self.vocab = []

        symbol_encoding = {
            ':)': 'xxsmilingfacexx',
            ':]': 'xxsmilingface2xx',
            ':-)': 'xxsmilingface3xx',
            ';)': 'xxwinksmilingfacexx',
            ':(': 'xxsadfacexx',
            ':-(': 'xxsadface2xx',
            ';(': 'xxcryingsadfacexx',
            ':|': 'xxblankfacexx',
            ':o': 'xxsurprisedfacexx',
            ':/': 'xxwryfacexx',
            ':D': 'xxgrinfacexx',
            ';D': 'xxwinkgrinfacexx',
            ':P': 'xxtonguefacexx',
            ':p': 'xxtongueface2xx',
            ';p': 'xxtongueface3xx',
            '...': 'xxthreedotsxx'
        }

        with open(root+filename, encoding='utf-8') as file:
            for line in file:
                word, *vector = line.split()
                self.vocab.append(word)
                word = word if word not in symbol_encoding else symbol_encoding[word]
                if word in self.tokenizer.word_index:
                    if self.tokenizer.word_index[word] < input_dim:
                        self.embedding_matrix[self.tokenizer.word_index[word]] = np.array(vector, dtype=np.float32)
        
        self.coverage = np.count_nonzero(np.count_nonzero(self.embedding_matrix, axis=1))/input_dim
        print(fr'coverage: {self.coverage :.4f}')
        
        return self.embedding_matrix

    def words_not_covered(self):
        indices = np.where(np.count_nonzero(self.embedding_matrix, axis=1) == 0)[0]
        indices = indices[indices != 0]
        words = np.array(list(self.tokenizer.word_index.keys()))[:self.input_dim]
        not_covered = tuple(zip(words[indices-1], indices))
        return not_covered


def preprocess_documents(docs, root='', twitter=False):
    glove_vocab = WordEmbeddings().get_glove_vocab(root=root, twitter=twitter)
    slang = pd.read_csv(root+'slang_no_duplicates_manual.csv')
    
    slang = slang[~slang['acronym'].isin(glove_vocab)].reset_index(drop=True)
    slang_mapping = dict(list(slang.to_records(index=False)))
    slang_mapping = {re.compile(r'\b{}\b'.format(k)): v for k, v in slang_mapping.items()}
    
    X = (docs
         .str.replace(r'http\S+', '', regex=True)
         .str.replace(r'\bu/\S+', '', regex=True)
         .str.replace(r'\:\)+(?!\S)', ' xxsmilingfacexx ', regex=True)
         .str.replace(r'\:\]+(?!\S)', ' xxsmilingface2xx ', regex=True)
         .str.replace(r'\:\-\)+(?!\S)', ' xxsmilingface3xx ', regex=True)
         .str.replace(r'\;\)+(?!\S)', ' xxwinksmilingfacexx ', regex=True)
         .str.replace(r'\:\(+(?!\S)', ' xxsadfacexx ', regex=True)
         .str.replace(r'\:\-\(+(?!\S)', ' xxsadface2xx ', regex=True)
         .str.replace(r'\;\(+(?!\S)', ' xxcryingsadfacexx ', regex=True)
         .str.replace(r'\:\|+(?!\S)', ' xxblankfacexx ', regex=True)
         .str.replace(r'\:o+(?!\S)', ' xxsurprisedfacexx ', regex=True)
         .str.replace(r'\:\/+(?!\S)', ' xxwryfacexx ', regex=True)
         .str.replace(r'\:D+(?!\S)', ' xxgrinfacexx ', regex=True)
         .str.replace(r'\;D+(?!\S)', ' xxwinkgrinfacexx ', regex=True)
         .str.replace(r'\:P+(?!\S)', ' xxtonguefacexx ', regex=True)
         .str.replace(r'\:p+(?!\S)', ' xxtongueface2xx ', regex=True)
         .str.replace(r'\;p+(?!\S)', ' xxtongueface3xx ', regex=True)
         .str.replace(r'...', ' xxthreedotsxx ', regex=False)
         .apply(lambda x: contractions.fix(x))
         .str.lower()
        )
    
    for abbr, expan in slang_mapping.items():
        present = X.str.contains(abbr)
        X[present] = X[present].str.replace(abbr, expan, regex=True)
    
    include_in_vocab = list(string.punctuation)

    include_in_vocab.remove('-')
    include_in_vocab.remove("'")

    include_in_vocab.append("'s")
    include_in_vocab.append("'ll")
    
    for symbol in include_in_vocab:
        X = X.str.replace(symbol, fr' {symbol} ', regex=False)

    # X = X.str.replace(r'\d+', '', regex=True)
    
    return X

def tokenize(data, input_dim=10000, filters='', quantile=0.9, verbose=False):
    tokenized_data = {}
    tokenizers = {}

    for which in ['comment', 'parent']:
        suffix = '' if which == 'comment' else '_par'

        tokenizer = Tokenizer(num_words=input_dim, filters=filters)
        tokenizer.fit_on_texts(data['X_train'+suffix])

        for x in ['X_train', 'X_val', 'X_test']:
            seq = tokenizer.texts_to_sequences(data[x+suffix])
            if x == 'X_train':
                maxlen = int(np.quantile([len(i) for i in seq], quantile))
                if verbose:
                    print(fr'{x+suffix}: maxlen = {maxlen}')
            tokenized_data[x+suffix] = pad_sequences(seq, padding='post', maxlen=maxlen)

        tokenizers[which] = tokenizer

    return tokenizers, tokenized_data