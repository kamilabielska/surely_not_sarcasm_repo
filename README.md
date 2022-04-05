# master thesis: sarcasm detection

### progress

|method|no context|context|
|------------|----------|-------|
|bag of words|◼◼◼  |◼ |
|RNN|◼◼◼  |◼ |
|CNN|◼◼◼  |◼◼     |
|RNN + CNN|◼◼◼  |◼ |
|attention|◼     | |
|transfer learning|◼◼     | |

◼ — started  
◼◼◼ — good for now  
◼◼◼◼◼ — finished, final version


tasks:
- keep implementing architectures from papers
- try different types of embeddings
- and embeddings dimensions
- play with hyperparameters

***

### plan
1. **introduction**
    - about sarcasm, sarcasm detection in nlp context
    - literature overview: what has been done a) with the use of machine learning and hand-crafted features, b) with the use of neural networks/deep learning, c) taking context into account
    - structure of the thesis
2. **theory**
    1. neural network
    2. CNN
    3. RNN
        - LSTM
        - GRU
    4. attention
    5. transformers?
    6. embeddings?
3. **experiments**
    1. data description
    2. text preprocessing
    3. proposed architectures
    4. results + error analysis
4. **conclusions**
