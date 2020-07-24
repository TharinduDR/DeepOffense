# DeepOffense : Multilingual Offensive Language Identification with Cross-lingual Embeddings

DeepOffense provides state-of-the-art models for multilingual offensive language identification.

## Installation
You first need to install PyTorch. THe recommended PyTorch version is 1.5.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, you can install from source by cloning the repository and running:

```bash
git clone https://github.com/TharinduDR/DeepOffense.git
cd DeepOffense
pip install -r requirements.txt
```

## Run the examples
Examples are included in the repository but are not shipped with the library. Please refer the [examples](/examples) directory for the examples. Each directory in the [examples](examples) folder contains different languages.


## Pretrained Models
Following pre-trained models are released. We will be keep releasing new models. Please keep in touch. 

| Language      | Objective |    Model           |  Model Link                          | Data                                                                         | Macro F1 | Weighted F1 | 
|:-------------:|-----------| ------------------:|:------------------------------------:|:----------------------------------------------------------------------------:| ------:  | ----------: |  
| Arabic        | OFF/NOT   | BERT-m             |                                      | [OffenseEval 2020](https://sites.google.com/site/offensevalsharedtask/)      |          |             |  
|               |           | BERT-m TL          |                                      |                                                                              |          |             | 
|               |           | XLM-R              |                                      |                                                                              |          |             | 
|               |           | XLM-R  TL          |                                      |                                                                              |          |             | 
| Bengali       |NAG/CAG/OAG| BERT-m             |                                      | [TRAC 2](https://sites.google.com/view/trac2/)                               |          |             |  
|               |           | BERT-m TL          |                                      |                                                                              |          |             | 
|               |           | XLM-R              |                                      |                                                                              |          |             | 
|               |           | XLM-R  TL          |                                      |                                                                              |          |             | 
| Danish        | OFF/NOT   | BERT-m             |                                      | [OffenseEval 2020](https://sites.google.com/site/offensevalsharedtask/)      |          |             |  
|               |           | BERT-m TL          |                                      |                                                                              |          |             | 
|               |           | XLM-R              |                                      |                                                                              |          |             | 
|               |           | XLM-R  TL          |                                      |                                                                              |          |             | 
| Greek         | OFF/NOT   | BERT-m             |                                      | [OffenseEval 2020](https://sites.google.com/site/offensevalsharedtask/)      |          |             |  
|               |           | BERT-m TL          |                                      |                                                                              |          |             | 
|               |           | XLM-R              |                                      |                                                                              |          |             | 
|               |           | XLM-R  TL          |                                      |                                                                              |          |             | 
| Hindi         | OFF/NOT   | BERT-m             |                                      | [HASOC 2019](https://hasocfire.github.io/hasoc/2019/index.html)              |          |             |  
|               |           | BERT-m TL          |                                      |                                                                              |          |             | 
|               |           | XLM-R              |  [Model.zip](https://bit.ly/2ZW5py9) |                                                                              |          |             | 
|               |           | XLM-R  TL          |                                      |                                                                              |          |             | 
| Malayalam     | OFF/NOT   | BERT-m             |                                      | [HASOC 2020](https://hasocfire.github.io/hasoc/2020/index.html)              |          |             |  
|               |           | BERT-m TL          |                                      |                                                                              |          |             | 
|               |           | XLM-R              |  [Model.zip](https://bit.ly/3eZ5Iga) |                                                                              |          |             | 
|               |           | XLM-R  TL          |                                      |                                                                              |          |             | 
| Sinhala       | Neutral/Sexism/Racist | BERT-m |                                     |[GitHub](https://github.com/renuka-fernando/sinhalese_language_racism_detection)|          |             |  
|               |           | BERT-m TL          |                                      |                                                                              |          |             | 
|               |           | XLM-R              |                                      |                                                                              |          |             | 
|               |           | XLM-R  TL          |                                      |                                                                              |          |             | 
| Spanish       | OFF/NOT   | BERT-m             |                                      | [HATEVAL 2020](https://competitions.codalab.org/competitions/19935)          |          |             |  
|               |           | BERT-m TL          |                                      |                                                                              |          |             | 
|               |           | XLM-R              |                                      |                                                                              |          |             | 
|               |           | XLM-R  TL          |                                      |                                                                              |          |             | 
| Turkish       | OFF/NOT   | BERT-m             |                                      | [OffenseEval 2020](https://sites.google.com/site/offensevalsharedtask/)      |          |             |  
|               |           | BERT-m TL          |                                      |                                                                              |          |             | 
|               |           | XLM-R              |                                      |                                                                              |          |             | 
|               |           | XLM-R  TL          |                                      |                                                                              |          |             | 

Once downloading them and unzipping it, they can be loaded easily

```bash
model = ClassificationModel("xlmroberta", "path,  use_cuda=torch.cuda.is_available())
```




## Citation
Please consider citing us if you use the library. 
```bash
Coming soon!
Please keep in touch
```