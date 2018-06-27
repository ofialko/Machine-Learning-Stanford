import pandas as pd
import numpy as np
from collections import Counter
import math, random
import matplotlib.pyplot as plt

def majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def clean_data():
    data = pd.read_csv('data.csv')
    y = data.diagnosis_result
    X = data.iloc[:,2:]
    # scaling and normalizing data
    X = (X - X.mean())/X.std()

    return X.values,y.values

if __name__ == "__main__":
    X,y = clean_data()
    # try different k
    for k in [1,3,5,7,9,11,13,15,17,19,21,23,25]:
        num_correct = 0
        for i in range(len(X)):
            sample_X = X[i]
            sample_y = y[i]

            other_X = np.delete(X,i,axis=0)
            other_y = np.delete(y,i)

            distances = np.sum((other_X - sample_X)**2,axis=1)
            d_sorted = sorted(zip(distances,other_y),key=lambda x: x[0])[:k]
            labels = [label for _,label in d_sorted]
            predicted_label = majority_vote(labels)

            if predicted_label == sample_y:
                num_correct += 1

        print(k, "neighbor[s]:", num_correct, "correct out of", len(X))
