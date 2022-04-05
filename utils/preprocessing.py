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

    def get_glove_vocab(self, emb_type='wikipedia', root=''):
        min_vec_len = {
            'wikipedia': 50,
            'twitter': 25,
            'crawl_uncased': 300,
            'crawl_cased': 300
        }

        path_to_file = fr'glove/{emb_type}/{min_vec_len[emb_type]}d.txt'
        self.vocab = []

        with open(root+path_to_file, encoding='utf-8') as file:
            for line in file:
                self.vocab.append(line.split()[0])

        return self.vocab

    def get_glove_embeddings(self, input_dim, vec_len, tokenizer, emb_type='wikipedia', root='', init='zeros'):
        self.tokenizer = tokenizer
        self.input_dim = input_dim

        if init == 'zeros':
            self.embedding_matrix = np.zeros((input_dim, vec_len))
        elif init == 'uniform':
            self.embedding_matrix = np.random.uniform(-0.05, 0.05, (input_dim, vec_len))
            self.embedding_matrix[0, :] = 0

        path_to_file = fr'glove/{emb_type}/{vec_len}d.txt'

        if emb_type != 'twitter':
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
                '<3': 'xxheartxx',
                '...': 'xxthreedotsxx'
            }
        else:
            symbol_encoding = {
                '<url>': 'xxurlxx',
                '<user>': 'xxuserxx',
                '<smile>': 'xxsmilexx',
                '<lolface>': 'xxlolfacexx',
                '<sadface>': 'xxsadfacexx',
                '<neutralface>': 'xxneutralfacexx',
                '<heart>': 'xxheartxx',
                '<hashtag>': 'xxhashtagxx',
                '<repeat>': 'xxrepeatxx',
                '<elong>': 'xxelongxx',
                '<allcaps>': 'xxallcapsxx',
                '<number>': 'xxnumberxx'
            }

        self.words_covered = []

        with open(root+path_to_file, encoding='utf-8') as file:
            for line in file:
                word, *vector = line.split()
                word = word if word not in symbol_encoding else symbol_encoding[word]

                if word in self.tokenizer.word_index:
                    word_id = self.tokenizer.word_index[word]
                    if word_id < input_dim:
                        self.embedding_matrix[word_id] = np.array(vector, dtype=np.float32)
                        self.words_covered.append(word_id)
        
        print(fr'coverage: {len(self.words_covered)/input_dim :.4f}')
        
        return self.embedding_matrix

    def words_not_covered(self):
        all_ids = np.arange(1, self.input_dim)
        not_covered_ids = all_ids[~np.isin(all_ids, self.words_covered)]
        words = np.array(list(self.tokenizer.word_index.keys()))[:self.input_dim]
        not_covered = tuple(zip(words[not_covered_ids-1], not_covered_ids))
        return not_covered


def preprocess_documents(docs, root='', emb_type='wikipedia'):
    glove_vocab = WordEmbeddings().get_glove_vocab(root=root, emb_type=emb_type)
    slang = pd.read_csv(root+'slang_no_duplicates_manual.csv')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs)
    
    slang = slang[~slang['acronym'].isin(glove_vocab)]
    slang = slang[slang['acronym'].isin(list(tokenizer.word_index.keys()))].reset_index(drop=True)
    slang_mapping = dict(list(slang.to_records(index=False)))
    slang_mapping = {re.compile(r'\b{}\b'.format(k)): v for k, v in slang_mapping.items()}

    # re.compile(r'{}+(?!\S)'.format(re.escape(':)')))
    
    if emb_type != 'twitter':
        X = (docs
             .str.replace(r'http\S+', '', regex=True)
             .str.replace(r'\bu/\S+', '', regex=True) # \w
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
             .str.replace(r'\<3+(?!\S)', ' xxheartxx ', regex=True)
             .str.replace(r'...', ' xxthreedotsxx ', regex=False)
            )
    else:
        eyes = "[8:=;]"
        nose = "['`\-]?"

        X = (docs
            .str.replace(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', 'xxurlxx', regex=True)
            .str.replace(r'\bu/\w+', 'xxuserxx', regex=True)
            .str.replace(fr'{eyes}{nose}[)d]+(?!\S)|[(]+{nose}{eyes}', ' xxsmilexx ', regex=True, case=False)
            .str.replace(fr'{eyes}{nose}p+(?!\S)', ' xxlolfacexx ', regex=True, case=False)
            .str.replace(fr'{eyes}{nose}\(+(?!\S)|\)+{nose}{eyes}', ' xxsadfacexx ', regex=True)
            .str.replace(fr'{eyes}{nose}[\/|l*]', ' xxneutralfacexx ', regex=True)
            .str.replace(r'\<3+(?!\S)', ' xxheartxx ', regex=True)
            .str.replace(r'#\S+', 'xxhashtagxx ', regex=True)
            .str.replace(r'([!?.]){2,}', r'\1 xxrepeatxx ', regex=True)
            .str.replace(r'\b(\S*?)(.)\2{2,}\b', r'\1\2 xxelongxx ', regex=True)
            .str.replace(r"(\b[^a-z0-9\W()<>'`\-]{2,}\b)", r'\1 xxallcapsxx ', regex=True)
            )

    X = X.apply(lambda x: contractions.fix(x)).str.lower()

    for abbr, expan in slang_mapping.items():
        present = X.str.contains(abbr)
        X[present] = X[present].str.replace(abbr, expan, regex=True)

    if emb_type == 'twitter':
        X = X.str.replace(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' xxnumberxx ', regex=True)
    
    include_in_vocab = list(string.punctuation)

    include_in_vocab.remove('-')
    include_in_vocab.remove("'")

    include_in_vocab.append("'s")
    include_in_vocab.append("'ll")
    
    for symbol in include_in_vocab:
        X = X.str.replace(symbol, fr' {symbol} ', regex=False)
    
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