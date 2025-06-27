<div align="justify">This subfolder contains the codes for NER-based text anonymization experiments. To run the codes, the dependencies need to be installed using the "install.sh" bash file. Fourteen pretrained models have been used. The pretrained models include:
  
- dslim/bert-base-NER
- dslim/bert-large-NER
- dbmdz/bert-base-cased-finetuned-conll03-english
- dbmdz/bert-large-cased-finetuned-conll03-english
- elastic/distilbert-base-uncased-finetuned-conll03-english
- Davlan/xlm-roberta-base-ner-hrl
- xlm-roberta-large-finetuned-conll03-english
- Jean-Baptiste/roberta-large-ner-english
- Davlan/xlm-roberta-large-ner-hrl
- Davlan/bert-base-multilingual-cased-ner-hrl
- spacy/en_core_web_sm
- spacy/en_core_web_md
- spacy/en_core_web_lg
- spacy/en_core_web_trf
    
The individual pretrained models need to be changed in line 144 e.g. "dbmdz/bert-base-cased-finetuned-conll03-english"as desired and the corresponding output directory in line 343 e.g. "". The entity categories in the pretrained models and the experimental dataset (CECILIA) have been unified. For information about and access to CECILIA dataset used, kindly consult this [page](https://gvis.unileon.es/datasets-cecilia-10c-900-ner/)</div>
