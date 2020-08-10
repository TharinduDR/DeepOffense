import os
import shutil
import time
import csv
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split
import emoji

from deepoffense.classification import ClassificationModel
from deepoffense.language_modeling.language_modeling_model import LanguageModelingModel
from examples.common.download import download_from_google_drive
from examples.common.evaluation import macro_f1, weighted_f1
from examples.common.label_converter import decode, encode
from examples.malayalam.malayalam_deepoffense_config import LANGUAGE_FINETUNE, TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, language_modeling_args, args, SEED, RESULT_FILE, DRIVE_FILE_ID, GOOGLE_DRIVE, \
    DEV_RESULT_FILE, DEMOJIZE
from examples.common.print_stat import print_information

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

train = pd.read_csv("examples/malayalam/data/ml-Hasoc-offensive-train.csv", sep='\t',
                    header=None, names=["labels", "text"], quoting=csv.QUOTE_NONE)
dev = pd.read_csv("examples/malayalam/data/ml-Hasoc-offensive-dev.csv", sep='\t',
                  header=None, names=["labels", "text"], quoting=csv.QUOTE_NONE)

test = pd.read_csv("examples/malayalam/data/ml-Hasoc-offensive-test.csv", sep=',',
                  header=None, names=["id", "text"], quoting=csv.QUOTE_NONE)

if DEMOJIZE:
    train['text'] = train['text'].apply(lambda x: emoji.demojize(x))
    dev['text'] = dev['text'].apply(lambda x: emoji.demojize(x))
    test['text'] = test['text'].apply(lambda x: emoji.demojize(x))

if LANGUAGE_FINETUNE:
    train_list = train['text'].tolist()
    dev_list = dev['text'].tolist()
    test_list = test['text'].tolist()
    complete_list = train_list + dev_list + test_list
    lm_train = complete_list[0: int(len(complete_list)*0.8)]
    lm_test = complete_list[-int(len(complete_list)*0.2):]

    with open(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), 'w') as f:
        for item in lm_train:
            f.write("%s\n" % item)

    with open(os.path.join(TEMP_DIRECTORY, "lm_test.txt"), 'w') as f:
        for item in lm_test:
            f.write("%s\n" % item)

    model = LanguageModelingModel("auto", MODEL_NAME, args=language_modeling_args, use_cuda=torch.cuda.is_available())
    model.train_model(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), eval_file=os.path.join(TEMP_DIRECTORY, "lm_test.txt"))
    MODEL_NAME = language_modeling_args["best_model_dir"]


# Train the model
print("Started Training")

train['labels'] = encode(train["labels"])
dev['labels'] = encode(dev["labels"])

dev_sentences = dev['text'].tolist()
dev_preds = np.zeros((len(dev), args["n_fold"]))

test_sentences = test['text'].tolist()
test_preds = np.zeros((len(test), args["n_fold"]))

train = pd.concat([train, dev])

if args["evaluate_during_training"]:
    for i in range(args["n_fold"]):
        if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
            shutil.rmtree(args['output_dir'])
        print("Started Fold {}".format(i))
        model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                    use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
        model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
        model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args,
                                    use_cuda=torch.cuda.is_available())

        dev_predictions, dev_raw_outputs = model.predict(dev_sentences)
        dev_preds[:, i] = dev_predictions

        test_predictions, test_raw_outputs = model.predict(test_sentences)
        test_preds[:, i] = test_predictions
        print("Completed Fold {}".format(i))
    # select majority class of each instance (row)
    final_dev_predictions = []
    for row in dev_preds:
        row = row.tolist()
        final_dev_predictions.append(int(max(set(row), key=row.count)))
    dev['predictions'] = final_dev_predictions

    final_test_predictions = []
    for row in test_preds:
        row = row.tolist()
        final_test_predictions.append(int(max(set(row), key=row.count)))
    test['predictions'] = final_test_predictions

else:
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    dev_predictions, raw_dev_outputs = model.predict(dev_sentences)
    dev['predictions'] = dev_predictions

    test_predictions, raw_test_outputs = model.predict(test_sentences)
    test['predictions'] = dev_predictions

dev['predictions'] = decode(dev['predictions'])
dev['labels'] = decode(dev['labels'])

test['label'] = decode(test['predictions'])

time.sleep(5)

print_information(dev, "predictions", "labels")
dev.to_csv(os.path.join(TEMP_DIRECTORY, DEV_RESULT_FILE),  header=True, sep='\t', index=False, encoding='utf-8')
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE),  header=True, sep=',', index=False, encoding='utf-8')

