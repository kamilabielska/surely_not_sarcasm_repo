import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    
def show_errors(model, X_not_token, X_token, y, n=5, X_parents=None):    
    y_pred = model.predict(X_token).round().flatten()

    fn_mask, fp_mask = (y != y_pred) & (y == 1), (y != y_pred) & (y == 0)
    rand_fn = np.random.randint(0, np.count_nonzero(fn_mask)-1, size=n)
    rand_fp = np.random.randint(0, np.count_nonzero(fp_mask)-1, size=n)

    fn_comments, fp_comments = X_not_token[fn_mask][rand_fn], X_not_token[fp_mask][rand_fp]
    if X_parents is not None:
        fn_parents, fp_parents = X_parents[fn_mask][rand_fn], X_parents[fp_mask][rand_fp]
    
    print('False negatives:')
    print('---------------------------')
    for i in range(n):
        if X_parents is None:
            print(fn_comments[i])
        else:
            print(fr'parent: {fn_parents[i]}')
            print(fr'comment: {fn_comments[i]}')
            print('')
    print('')
    print('False positives:')
    print('---------------------------')
    for i in range(n):
        if X_parents is None:
            print(fp_comments[i])
        else:
            print(fr'parent: {fp_parents[i]}')
            print(fr'comment: {fp_comments[i]}')
            print('')
