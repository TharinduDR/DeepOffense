from sklearn.metrics import recall_score, precision_score, f1_score

labels = ['Offensive', 'Not_offensive']


def print_information(df, pred_column, real_column):
    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    print("Stat of the Offensive Class")
    print("Recall {}".format(recall_score(real_values, predictions, labels=labels, pos_label='Offensive')))
    print("Precision {}".format(precision_score(real_values, predictions, labels=labels, pos_label='Offensive')))
    print("F1 Score {}".format(f1_score(real_values, predictions, labels=labels, pos_label='Offensive')))

    print("Stat of the Not Offensive Class")
    print("Recall {}".format(recall_score(real_values, predictions, labels=labels, pos_label='Not_offensive')))
    print("Precision {}".format(precision_score(real_values, predictions, labels=labels, pos_label='Not_offensive')))
    print("F1 Score {}".format(f1_score(real_values, predictions, labels=labels, pos_label='Not_offensive')))

    print("===============================")

    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighter F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))

    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))