# DT2119-Proj
DT2119 Speech and Speaker Recognition Project

## Objective
Semi-supervised learning on frame-based phoneme classification

## Tasks
- [ ] literatal study
- [x] Proof-of-concept (Bi-LSTM, LSTM, autoencoder)
- [x] Set the project objective
- [x] Dataset selection
- [x] Experiement design
- [x] Feature extraction (closly related to labs)
- [ ] Data loader and exp code
- [ ] Model code implementations
- [ ] Poster
- [ ] Final Report and discussion

## Report infomation
Link to report:
https://www.overleaf.com/6637372248xzbwwqpjpdzj

The peer review score is closely related to the course objective.
Please read it on:
https://www.kth.se/student/kurser/kurs/DT2119?l=en

## Dataset
1. [TIMIT](https://github.com/philipperemy/timit)
2. TIDIGIT (may not be able to use)

### TIMIT Dataset
See section 3.4.3 for more info [doc](https://perso.limsi.fr/lamel/TIMIT_NISTIR4930.pdf)

#### Training and validation 
Training + validation = 330 SX texts * 7 speakers + 1386 SI texts  
validation = 184 sentences (unknown how to get it, random?)  
Therefore, training = 3696 - 184 = 3152 sentances

#### Testing
Use the standard core data set = 192 sentences. Splited by DARPA-ISTO.

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