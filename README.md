Speech Recognition Final project

## Set Up
```
wget https://mm.kaist.ac.kr/share/Anaconda3-2021.11-Linux-x86_64.sh --no-check-certificate
sh Anaconda3-2021.11-Linux-x86_64.sh
conda create --name sr --file sr.yml
conda activate sr
```
## GOAL
Train an under 15M Korean ASR model on KsponSpeech dataset only.

## Model Structure
**Originally**
- Raw Audio -> MFCC(40) -> CNN -> bi-LSTM -> Linear -> Pred Character
- CNNs
  - (Dropout - Conv1D - BatchNorm1D - ReLU) X 2

**Proposed**
- Raw Audio -> MFCC(40/80) -> Conformer -> bi-LSTM -> Linear -> Pred Character

## To Do
- [x] Decide the Conformer Architecture : How many layers, Decoder Structure ...
- [x] KenLM training 
- [x] Joint Decoding With KenLM
