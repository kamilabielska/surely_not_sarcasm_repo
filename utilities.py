import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import contractions
import string
import re
import time
import datetime
import copy
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

def plot_history(fit):
    fig, axes = plt.subplots(1, 2, figsize=(14,4))

    for i, which in enumerate(['accuracy', 'loss']):
        ax = axes[i]
        ax.plot(fit.history[which], label=which)
        ax.plot(fit.history['val_'+which], label='val_'+which)
        ax.set_xlabel('epoch')
        ax.set_ylabel(which)
        ax.legend();
        
def evaluate(model, X, y):
    y_pred = model.predict(X).round().flatten()
    metrics = [
        fr'acc = {accuracy_score(y, y_pred) :.4f}',
        fr'prec = {precision_score(y, y_pred) :.4f}',
        fr'rec = {recall_score(y, y_pred) :.4f}',
        fr'f1 = {f1_score(y, y_pred) :.4f}'
    ]

    fig, ax = plt.subplots(figsize=(5,5));
    confusion = confusion_matrix(y, y_pred, normalize='true')
    matrix_display = ConfusionMatrixDisplay(confusion, display_labels=['non-sarcastic', 'sarcastic'])
    matrix_display.plot(colorbar=False, ax=ax)
    plt.grid(False)
    plt.text(1.6, 0.65, '\n'.join(metrics), fontsize=15);
    
def show_errors(model, X_not_token, X_token, y, n=3, maxlen=20):    
    y_pred = model.predict(X_token).round().flatten()
    false_negatives = X_not_token[(y != y_pred) & (y == 1)] # sarcastic comments classified as non-sarcastic
    false_positives = X_not_token[(y != y_pred) & (y == 0)] # non-sarcastic comments classified as sarcastic
    rand_fn = np.random.randint(0, false_negatives.shape[0]-1, size=n)
    rand_fp = np.random.randint(0, false_positives.shape[0]-1, size=n)

    print('False negatives:')
    print('---------------------------')
    for sentence in false_negatives[rand_fn].tolist():
        print(sentence)
    print('')
    print('False positives:')
    print('---------------------------')
    for sentence in false_positives[rand_fp].tolist():
        print(sentence)


def train_model(model, criterion, optimizer, scheduler, epochs, dataloaders, data_size, device, es_patience=None, rep=100):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    epochs_no_improvement = 0

    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        print('')
        print(fr'======== Epoch {epoch + 1} / {epochs} ========')
        print('Training...')
        t0 = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()
            
            dataloader = dataloaders[phase]
            running_loss = 0
            running_corrects = 0

            for step, batch in enumerate(dataloader):
                if phase == 'train' and step % rep == 0 and not step == 0:
                    print(fr'Batch {step:>5,} of {len(dataloader):>5,}. Elapsed: {(time.time()-t0):.2f} s.')

                ids = batch[0].to(device)
                masks = batch[1].to(device)
                labels = batch[2].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(ids, attention_mask=masks, return_dict=False)
                    _, preds = torch.max(outputs[0], 1)
                    loss = criterion(outputs[0], labels)
            
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()*ids.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss/data_size[phase]
            epoch_accuracy = running_corrects.double()/data_size[phase]
            
            history[phase + '_loss'].append(epoch_loss)
            history[phase + '_accuracy'].append(epoch_accuracy)

            print(fr'== {phase} == loss: {epoch_loss:.4f} accuracy: {epoch_accuracy:.4f}')

            if phase == 'val':
                if epoch_accuracy > best_acc:
                    best_acc = epoch_accuracy
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improvement = 0
                else:
                    epochs_no_improvement += 1
        
        if es_patience is not None and epochs_no_improvement >= es_patience:
            print('')
            print(fr'No validation accuracy improvement for {es_patience} consecutive epochs. Stopping training...')
            break
    
    print('')
    print(fr'Training finished. Best validation accuracy: {best_acc:.4f}.')
    model.load_state_dict(best_model_wts)
    return model, history

def test_model(model, dataloader, data_size, device):
    model.eval()
    
    pred_labels, test_labels = [], []
    running_corrects = 0
    
    for batch_idx, (ids, masks, labels) in enumerate(dataloader):

        ids, labels, masks = [x.to(device) for x in [ids, labels, masks]] 
        
        with torch.no_grad():
            outputs = model(ids, attention_mask=masks, return_dict=False)
            _, preds = torch.max(outputs[0], 1)

        pred_labels += preds.tolist()
        test_labels += labels.data.tolist()

        running_corrects += torch.sum(preds == labels.data)
        
    accuracy = running_corrects.double()/data_size
    print(fr'test accuracy: {accuracy:.4f}')
    
    return np.array(pred_labels), np.array(test_labels)