import numpy as np
import pandas as pd
import contractions
import string
import re

from keras.preprocessing.text import Tokenizer


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

    def get_glove_vectors(self, tokens, vec_len, emb_type='wikipedia', root=''):

        path_to_file = fr'glove/{emb_type}/{vec_len}d.txt'
        vectors = {}

        with open(root+path_to_file, encoding='utf-8') as file:
            for line in file:
                word, *vector = line.split()
                if word in tokens:
                    vectors[word] = np.array(vector, dtype=np.float32)
                    if len(tokens) == len(vectors.keys()):
                        break

        return vectors

    def get_glove_embeddings(self, input_dim, vec_len, doc_vocab, emb_type='wikipedia', root=''):
        self.doc_vocab = doc_vocab
        self.input_dim = input_dim

        self.embedding_matrix = np.zeros((input_dim, vec_len))

        path_to_file = fr'glove/{emb_type}/{vec_len}d.txt'
        self.words_covered = []

        with open(root+path_to_file, encoding='utf-8') as file:
            for line in file:
                word, *vector = line.split()
                if word in doc_vocab:
                    word_id = doc_vocab.index(word)
                    if word_id < input_dim:
                        self.embedding_matrix[word_id] = np.array(vector, dtype=np.float32)
                        self.words_covered.append(word_id)
        
        print(fr'coverage: {len(self.words_covered)/input_dim :.4f}')
        
        return self.embedding_matrix

    def words_not_covered(self):
        all_ids = np.arange(self.input_dim)
        not_covered_ids = all_ids[~np.isin(all_ids, self.words_covered)]
        not_covered = tuple(zip(np.array(self.doc_vocab)[not_covered_ids], not_covered_ids))
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
    
    X = (docs
        .str.replace(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', 'xxurlxx', regex=True)
        .str.replace(r'\bu/\w+', 'xxuserxx', regex=True)
        )

    if emb_type != 'twitter':
        token_encoding = {
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

        for token, encoding in token_encoding.items():
            if token == '...':
                X = X.str.replace(token, fr' {encoding} ', regex=False)
            else:
                pattern = re.compile(r'{}+(?!\S)'.format(re.escape(token)))
                X = X.str.replace(pattern, fr' {encoding} ', regex=True)
    else:
        eyes = "[8:=;]"
        nose = "['`\-]?"

        token_encoding = {
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

        X = (X
            .str.replace(fr'{eyes}{nose}[)d]+(?!\S)|[(]+{nose}{eyes}', ' xxsmilexx ', regex=True, case=False)
            .str.replace(fr'{eyes}{nose}p+(?!\S)', ' xxlolfacexx ', regex=True, case=False)
            .str.replace(fr'{eyes}{nose}\(+(?!\S)|\)+{nose}{eyes}', ' xxsadfacexx ', regex=True)
            .str.replace(fr'{eyes}{nose}[\/|l*]', ' xxneutralfacexx ', regex=True)
            .str.replace(r'\<3+(?!\S)', ' xxheartxx ', regex=True)
            .str.replace(r'#\S+', ' xxhashtagxx ', regex=True)
            .str.replace(r'([!?.]){2,}', r'\1 xxrepeatxx ', regex=True)
            )

    X = (X
        .apply(lambda x: contractions.fix(x, slang=False))
        .str.replace(r'\b([a-zA-Z]+?)(.)\2{2,}\b', r'\1\2 xxelongxx ', regex=True)
        .str.replace(r"(\b[^a-z0-9\W()<>'`\-_]{2,}\b)", r'\1 xxallcapsxx ', regex=True)
        .str.lower()
        )

    for abbr, expan in slang_mapping.items():
        X = X.str.replace(abbr, expan, regex=True)

    if emb_type == 'twitter':
        X = X.str.replace(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' xxnumberxx ', regex=True)
    else:
        has_numbers = X.str.contains(r'\d+', regex=True)
        X[has_numbers] = X[has_numbers].str.replace(r'[-+]?[.,\d]*[\d]+[:,.]+[\d]+', ' xxnumberxx ', regex=True)
    
    include_in_vocab = list(string.punctuation)

    include_in_vocab.remove('-')
    include_in_vocab.remove("'")
    
    for symbol in include_in_vocab:
        X = X.str.replace(symbol, fr' {symbol} ', regex=False)

    X = X.str.replace(r"([a-zA-Z]+)'s", r"\1 's ", regex=True)
    X = X.str.replace(r"([a-zA-Z]+)'ll", r"\1 'll ", regex=True)
    X = X.str.replace(r"([a-zA-Z]+)'d", r"\1 'd ", regex=True)

    X = X.str.replace("'", " ' ", regex=False)

    for token, encoding in token_encoding.items():
        X = X.str.replace(encoding, token, regex=False)
    
    return X