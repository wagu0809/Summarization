# Summarization
Summarization model using Bert

This is a general framework of extraction summarization.

For now, not only BERT can be used to obtain SOTA results, also other new models like XLNet, XLM, RoBERTa

This is frame which can be easily changed to fit new models, data pre-processing can also be modified to fit to the model

in ./data, there are pre-processed CNN/daily mail data

in ./model, checkpoints saved during training 

in ./logs, train_log, validate_log and test_log are saved

Pre-processed data(https://drive.google.com/open?id=1DN7ClZCCXsk2KegmC6t4ClBwtAf5galI)

Data description:

Different with BERT, to represent each individual sentence, [CLS] token is added to the beginning and [SEP] to the end of each sentence and also segment embeddings, position embeddings are added.

Each [CLS] token collects the feature of the sentence next to it.
[![20191004004556.png](https://i.postimg.cc/c4FZYChF/20191004004556.png)](https://postimg.cc/9rwvHWv9)

Some summarization results of other works:
(http://nlpprogress.com/english/summarization.html)

