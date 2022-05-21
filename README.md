# master thesis: sarcasm detection

### description of the models

modele, które potencjalnie znajdą się w pracy, wszystkie zebrałam w pliku *all_the_models.ipynb*, oprócz bag-of-words w pliku *bag_of_words.ipynb* i DistilBERTa w pliku *transfer_learning.ipynb*, bo są trochę inne niż inne; poniższe nazwy odpowiadają sekcjom w notebooku, więc wystarczy wyszukać po nazwie

w nawiasie podana dokładność dla tego jednego runa, tak żeby mieć szybki podgląd na sytuację (tyle, że DistilBERT na validacyjnym, nie testowym)

\* w bag-of-words ostatnio coś zmieniałam i nie przystosowałam regularyzacji, w rzeczywistości będzie mniejsza
\*\* ostateczna ewaluacja będzie dla train-val-test splitu 70-15-15

**content-based (comment)**:
 - bag-of-words (0.6855): (właściwie bag-of-uni/bi/trigrams) baseline + univariate feature selection
 - GRU (0.7464): najprostsze z najprostszych, jedna warstwa GRU i tyle
 - bi-GRU (0.7489): żeby zobaczyć czy spojrzenie na komentarz z obydwu stron coś daje
 - bi-GRU + auxiliary features (0.7486): zastoswanie deep learningu miało mieć tą zaletę, że hand-crafted features miały być niepotrzebne, ale widziałam te artykuły, których autorzy i tak jakieś dodatkowe atrybuty dodawali i też to poprawiało wynik, więc to sprawdziłam
 - bi-GRU + emotion embeddings (0.7482): normalne GRU na komentarzu + embeddingi dla każdego ze słów stworzone na podstawie podobieństwa wektorów z wektorami "seed words" związanych z emocjami (np. anger, surprise) — podejście zainspirowane jednym z artykułów
 - CNN (0.7439): kilka warstw konwolucyjnych
 - CNN-LSTM-DNN (0.7447): oparte na jednym z artykułów, żeby zobaczyć jak sobie radzi w porównaniu z poprzednimi; ogólnie też eksperymentowałam trochę z połączeniem CNN i RNN, ale nie dawało to lepszych wyników niż samo RNN
 - MHA: self-attention (0.7448/0.7474): multi-head attention w połączeniu z GRU, w literaturze motywowane wykrywaniem kontrastu, niezgodności w sarkastycznych wypowiedziach, ale tutaj nie dało to właściwie nic, a nawet w tym pojedynczym runie wyszło gorzej niż samo RNN; wyplotowałam też przykładowe attention scores
 - transfer learning (0.7673): DistilBERT, to jeszcze nie zupdatowane z ostatnią wersją preprocessingu
 
**context-based (comment + parent comment, dwa osobne inputy do modelu)**:
 - bag-of-words (0.6892): baseline
 - GRU + GRU (0.7524): i komentarz i parent przechodzą przez jedną warstwę GRU, odpowiedznik GRU z content-based
 - CNN + CNN (0.7459): podobnie, odpowiednik CNN z content-based
 - bi-GRU + CNN (0.7508): comment -> bi-GRU, parent comment -> CNN
 - MHA: comment + parent (0.7487): analogicznie do content-based, tylko, że attention pomiędzy comment i parentem
 - MHA: emotion comment + parent (0.7503): tak jak wyżej, ale do attention wchodzą emotion embeddings
 - transfer learning (0.7813): DistilBERT — można dać dwa inputy, wtedy je oddziela specjalnym tokenem, także jeszcze nie zupdatowane
 
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


