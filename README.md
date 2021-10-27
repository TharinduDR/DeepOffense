# DeepOffense : Multilingual Offensive Language Identification with Transformers

DeepOffense provides state-of-the-art transformer models for multilingual offensive language identification. 

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
English offensive language detection pre-trained model trained with XLM-R  large model on OffensEval data can be downloaded using this [link](https://drive.google.com/file/d/1_P3dCLcN3XoJT8BRgFhrwdVMODgyejwI/view?usp=sharing).

Once downloading it and unzipping it, they can be loaded easily. To see how to begin the training process please refer the [examples](/examples) directory

```bash
model = ClassificationModel("xlmroberta", "path",  use_cuda=torch.cuda.is_available())
```




## Citation
Please consider citing us if you use the library. 
```bash
@inproceedings{ranasinghe-etal-2020-multilingual,
    title = "Multilingual Offensive Language Identification with Cross-lingual Embeddings",
    author = "Ranasinghe, Tharindu  and
      Zampieri, Marcos,
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov
    year = "2020",
    }
```

Citation for the Malayalam specific paper, 

```bash
@inproceedings{ranasinghe-etal-2020-wlv,
     title={WLV-RIT at HASOC 2020: Offensive Language Identification in Code-switched Texts},
      author={Ranasinghe, Tharindu and Zampieri, Marcos},
      year={2020},
      booktitle={Proceedings of FIRE}
}
```