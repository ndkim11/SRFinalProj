Speech Recognition Final project

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
- [ ] Decide the Conformer Architecture : How many layers, Decoder Structure ...
- [ ] KenLM training 
- [ ] Joint Decoding With KenLM
