# SLK-NER

The implementation of SEKE 2020 paper "[SLK-NER: Exploiting Second-order Lexicon Knowledge for Chinese NER]()"


# Requirements
python == 3.6.2<br>
torch == 1.1.0<br>


# How to use
  ### Dataset


  The original datasets can be found at [OntoNotes](https://catalog.ldc.upenn.edu/LDC2011T03), 
  [Weibo](https://github.com/hltcoe/golden-horse) and [Resume](https://github.com/jiesutd/LatticeLSTM/tree/master/ResumeNER).


  ### Input format:

  BMES tag scheme, with each character its label for one line. Sentences are splited with a null line.
  
  ```cpp
    美   B-LOC  
    国   E-LOC  
    的   O  
    华   B-PER  
    莱   M-PER  
    士   E-PER  
    
    我   O  
    跟   O  
    他   O  
    谈   O  
    笑   O  
    风   O  
    生   O   
  ``` 

  ### Pretrained Embeddings
  Character embeddings: [chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)  
  Word embeddings: [ctb.50d.vec](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing)  
  

  ### Training/Testing

  ```
  bash run_ner.sh
  ```
  
  
# Citation
```
@inproceedings{hu2020SLK-NER
author={Dou Hu and Lingwei Wei},
title={SLK-NER: Exploiting Second-order Lexicon Knowledge for Chinese NER},
journal={The 32nd International Conference on Software & Knowledge Engineering},
year={2020}
}
```
