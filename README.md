# English-2-French translator
Transformer neural network (seq2seq) model from scratch using pytorch for language translation from english to french.

## Warning
At this moment, the model is overfitting because it was trained on a small subset of the data. This is pure educational
project and my resources for training are limited (model was trained on a laptop with gtx 1050ti).
For example: "Resumption of the session" translates to "Reprise de la sessione" or similiar which is quite satisfactory, considering 
character level tokenization for both english and french languages.


## File description
- model.py: Pytorch model definition
- train.py: Training loop. Execute it to train defined number of epochs
- generate.py: To generate a french sentence, pass the input sentence to the generate() function
