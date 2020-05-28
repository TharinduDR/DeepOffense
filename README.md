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
Examples are included in the repository but are not shipped with the library.
*TL indicates Transfer Learning on English 
### Hindi

| Model    |   Macro F1    |  Weighted F1 |
|----------|--------------:|-------------:|
| BERT-m   |    0.8025     |   0.8030     |
| BERT-m (TL) |    0.8211  |   0.8220     |
| XLM-R    | 0.8061        |   0.8072     |
| XLM-R  (TL) | 0.8568        |   0.8580     |