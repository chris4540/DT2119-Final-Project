# DT2119-Proj
DT2119 Speech and Speaker Recognition Project

## Objective
Semi-supervised learning on frame-based phoneme classification

## Tasks
- [ ] literatal study
- [ ] Proof-of-concept (Bi-LSTM, LSTM, autoencoder)
- [ ] Set the project objective
- [x] Dataset selection
- [ ] Experiement design
- [ ] Feature extraction (closly related to labs)
- [ ] Model code implementations
- [ ] Poster
- [ ] Report and discussion

## Novelty
These points are probabily used but not limited to
1. Add two to three layers to the encoder, decoder, and the classifier
2. use ReLU/ week ReLU as the activation function
3. Add batch-normalization when the number of layers increased
4. compare different combinations of these points

## Report infomation
Link to report:
https://www.overleaf.com/6637372248xzbwwqpjpdzj

The peer review score is closely related to the course objective.
Please read it on:
https://www.kth.se/student/kurser/kurs/DT2119?l=en

## Dataset
1. [TIMIT](https://github.com/philipperemy/timit)
2. TIDIGIT (may not be able to use)
3. TODO

### Reference
1. [Akash Kumar Dhaka, 2017](http://www.speech.kth.se/glu2017/papers/GLU2017_paper_5.pdf)

2. [amazon's paper](https://arxiv.org/pdf/1904.01624.pdf)

3. http://www.nada.kth.se/~ann/exjobb/shiping_song.pdf

### Impl. reference
0. [Preprocessing] (https://github.com/Faur/TIMIT)
1. [Sparse autoencoder](https://github.com/Abhipanda4/Sparse-Autoencoders)
2. [This repo has examples on how to use librosa on speech and TIMT classification on Phoneme](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)

3. [Combining LSTM and Convolution Neural Networks for Phoneme Recognition](https://github.com/Pierre28/DT2119_Project)

https://github.com/awni/speech

https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders


https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/43905.pdf

https://github.com/snap-stanford/GraphRNN/blob/master/train.py

https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/