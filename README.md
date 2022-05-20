# master thesis: sarcasm detection

### description of the models

modele, które potencjalnie znajdą się w pracy, wszystkie zebrałam w pliku *all_the_models.ipynb*, oprócz bag-of-words w pliku *bag_of_words.ipynb* i DistilBERTa w pliku *transfer_learning.ipynb*, bo są trochę inne niż inne; poniższe nazwy odpowiadają sekcjom w notebooku, więc wystarczy wyszukać po nazwie:

**content-based (comment)**:
 - bag-of-words: (właściwie bag-of-uni/bi/trigrams) baseline + univariate feature selection
 - GRU (wynik): najprostsze z najprostszych, jedna warstwa GRU i tyle
 - bi-GRU: żeby zobaczyć czy spojrzenie na komentarz z obydwu stron coś daje
 - bi-GRU + auxiliary features: zastoswanie deep learningu miało mieć tą zaletę, że hand-crafted features miały być niepotrzebne, ale widziałam te artykuły, których autorzy i tak jakieś dodatkowe atrybuty dodawali i też to poprawiało wynik, więc to sprawdziłam
 - bi-GRU + emotion embeddings: normalne GRU na komentarzu + embeddingi dla każdego ze słów stworzone na podstawie podobieństwa wektorów z wektorami "seed words" związanych z emocjami (np. anger, surprise) — podejście zainspirowane jednym z artykułów
 - CNN: kilka warstw konwolucyjnych
 - CNN-LSTM-DNN: oparte na jednym z artykułów, żeby zobaczyć jak sobie radzi w porównaniu z poprzednimi; ogólnie też eksperymentowałam trochę z połączeniem CNN i RNN, ale nie dawało to lepszych wyników niż samo RNN
 - MHA: self-attention: multi-head attention w połączeniu z GRU, w literaturze motywowane wykrywaniem kontrastu, niezgodności w sarkastycznych wypowiedziach, ale tutaj nie dało to właściwie nic, a nawet w tym pojedynczym runie wyszło gorzej niż samo RNN; wyplotowałam też przykładowe attention scores
 - transfer learning: DistilBERT, to jeszcze nie zupdatowane z ostatnią wersją preprocessingu
 
**context-based (comment + parent comment, dwa osobne inputy do modelu)**:
 - bag-of-words: baseline
 - GRU + GRU: i komentarz i parent przechodzą przez jedną warstwę GRU, odpowiedznik GRU z content-based
 - CNN + CNN: podobnie, odpowiednik CNN z content-based
 - bi-GRU + CNN: comment -> bi-GRU, parent comment -> CNN
 - MHA: comment + parent: analogicznie do content-based, tylko, że attention pomiędzy comment i parentem
 - MHA: emotion comment + parent: tak jak wyżej, ale do attention wchodzą emotion embeddings
 - transfer learning: DistilBERT — można dać dwa inputy, wtedy je oddziela specjalnym tokenem, także jeszcze nie zupdatowane
 
folder *utils* zawiera pomocnisze funkcje do preprocessingu, ewaluacji i treningu w pytorchu (bo transfer learning mam w pytorchu, reszta w kerasie)

***

### progress

|method|no context|context|
|------------|----------|-------|
|bag of words|◼◼◼◼  |◼◼◼◼ |
|RNN|◼◼◼◼  |◼◼◼◼ |
|CNN|◼◼◼◼  |◼◼◼◼     |
|RNN + CNN|◼◼◼◼  |◼◼◼◼ |
|attention|◼◼◼◼     |◼◼◼◼ |
|transfer learning|◼◼◼◼     |◼◼◼◼ |

◼ — started  
◼◼◼ — good for now  
◼◼◼◼◼ — finished, final version


