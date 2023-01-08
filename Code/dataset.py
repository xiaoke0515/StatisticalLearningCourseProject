import numpy as np
import csv

train_features = np.load('dataset/train_features.npy')
train_labels = np.load('dataset/train_labels.npy')
test_features = np.load('dataset/test_features.npy')

def write_results(labels, file_name = 'result.csv'):
    headers = ['Id', 'Category']
    value = []
    for i, lab in enumerate(labels):
        value.append({'Id': i, 'Category': lab})
    with open(file_name,'w',encoding='utf-8',newline='') as fp:
        writer = csv.DictWriter(fp,headers)
        writer.writeheader()
        writer.writerows(value)