import os
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from deepoffense.classification import MultiLabelClassificationModel
from examples.common.evaluation import macro_f1, weighted_f1
from examples.german.german_deepoffense_config import TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, args, SEED, RESULT_FILE


if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

data = pd.read_csv('examples/german/data/GermEval21_Toxic_Train.csv')
test_data = pd.read_csv('examples/german/data/GermEval21_Toxic_TestData.csv')

training_instances = []
for index, row in data.iterrows():
    training_instance = [row['comment_text'], [row['Sub1_Toxic'], row['Sub2_Engaging'], row['Sub3_FactClaiming']]]
    training_instances.append(training_instance)

testing_ids = []
testing_texts = []
for test_index, test_row in test_data.iterrows():
    testing_ids.append(test_row["comment_id"])
    testing_texts.append(test_row["c_text"])

train = pd.DataFrame(training_instances)
train.columns = ["text", "labels"]

train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)

model = MultiLabelClassificationModel(MODEL_TYPE, MODEL_NAME, num_labels=3,  args=args, use_cuda=torch.cuda.is_available())
model.train_model(train_df)

# model = MultiLabelClassificationModel(MODEL_TYPE, args["best_model_dir"], num_labels=3,  args=args, use_cuda=torch.cuda.is_available())

# Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)
predictions, raw_outputs = model.predict(testing_texts)

print(predictions)
prediction_instances = []
for id, prediction in zip(testing_ids, predictions):
    print(prediction)
    print(type(prediction))
    prediction_instances.append([id, prediction[0], prediction[1], prediction[2]])


predictions = pd.DataFrame(prediction_instances)
predictions.columns = ["comment_id", "Sub1_Toxic", "Sub2_Engaging", "Sub3_FactClaiming"]

predictions.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE),  header=True, sep=',', index=False, encoding='utf-8')


