import sklearn
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import csv

from sklearn.preprocessing import LabelEncoder

from deepoffense.classification.classification_model import ClassificationModel
from examples.english.english_deepoffense_config import global_args

model = ClassificationModel("bert", "bert-base-multilingual-cased",  use_cuda=torch.cuda.is_available(),
                           args=global_args)

data = pd.read_csv('examples/english/data/olid-training-v1.0.tsv', sep="\t",  quoting=csv.QUOTE_NONE)
data = data.rename(columns={'tweet': 'text', 'subtask_a': 'label'}).dropna()
data = data[['text', 'label']]

le = LabelEncoder()
data['label'] = le.fit_transform(data["label"])

train, test = train_test_split(data, test_size=0.2)
train, eval_df = train_test_split(train, test_size=0.1)

model.train_model(train, eval_df=eval_df, f1=sklearn.metrics.f1_score, accuracy=sklearn.metrics.accuracy_score)
model = ClassificationModel("xlmroberta", global_args["best_model_dir"],
                           use_cuda=torch.cuda.is_available(), args=global_args)


result, model_outputs, wrong_predictions = model.eval_model(test, f1=sklearn.metrics.f1_score, accuracy=sklearn.metrics.accuracy_score)
print(result)