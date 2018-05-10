#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:28:22 2018

@author: jayborkar
"""

import pandas as pd

# Loading the dataset
df = pd.read_json("Digital_Music_5.json", lines=True)

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[['reviewerID', 'asin', 'overall']], reader)
s
#Spliting the dataset with 80/20 ratio
trainset,testset = train_test_split(data, test_size=.20)

#Importing Surprise Package and other library
from surprise import SVDpp
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import ndcg_score


def precision_recall(predictions, k=10, threshold=3.5):
    '''Return precision and recall for each user.'''

    # First map the predictions to each user.z
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est,_ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
   
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
        
    return top_n


# Use the famous SVD algorithm.
algo = SVDpp()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
#testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])


precisions, recalls = precision_recall(predictions, k=10, threshold=3.5)


accuracy.rmse(predictions)
accuracy.mae(predictions)

#Precision and recall can then be averaged over all users
preci=sum(prec for prec in precisions.values()) / len(precisions)
print("Precision",preci)

rec=sum(rec for rec in recalls.values()) / len(recalls)
print("Recall",rec)
#F-measure: F=2*Precision*Recall / (Precision + Recall)
Fm=2*preci*rec/(preci+rec)
print(Fm)

#print(ndcg_score(testset, predictions, k=2))
