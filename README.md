# Large-scale-Data-Analytics-in-Recommendation-Systems

E-commerces, such as Amazon or Ebay are putting a lot of money into Recommender systems. They are building great teams just to focus on improving the accuracy of their recommenders, because by doing so, users are much more tempted to buy more things.

**Goal:** Build a recommendation system with better accuracy to predict the ratings, users would give for products which they haven’t used before and to create a recommendation list for each user.

A simple strategy to create such a recommendation list for a user is to predict the ratings on all the items that user didn’t buy before, then rank the items in descending order of the predicted rating, and finally take the top items as the recommendation list.


# Dataset and Preprocessing

Researchers from UC San Diego released a complete Amazon dataset that is publicly available online (​http://jmcauley.ucsd.edu/data/amazon/)​. This dataset contains user reviews (numerical rating and textual comment) towards amazon products on 24 product categories, and there is an independent dataset for each product category. 

- Used the “Small subsets for experiment” (the 5-core dataset) on the website, which can be downloaded directly from the website.

- Selected ​Digital Music dataset from the set of available datasets. This dataset has 64,706 reviews and ratings.

- Out of all the attributes in the dataset, we were interested only in the productID, userID and rating for each review. Then, the data is split into train (80%) and test (20%).

# Approach - I

**Surprise**

Surprise​ is a Python ​scikit​ building and analyzing recommender systems. Surprise​ is a Python ​scikit​ building and analyzing recommender systems.

- Used matrix factorization-based algorithms such as SVD and SVD++ for predicting the ratings and recommending items.

- Evaluation of SVD: Train RMSE = 0.9196,  MAE = 0.6902

- Evaluation of SVD++: Train RMSE = 0.9076,  MAE = 0.6702


# Approach - II

**ALS (Alternating Least Squares)**

Collaborative filtering is commonly used for recommender systems. These techniques aim to fill in the missing entries of a user-item association matrix. spark.mllib currently supports model-based collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries. spark.mllib uses the alternating least squares (ALS) algorithm to learn these latent factors.


The dataset was split in 80:20, with 80% training and 20% testing sets. The ALS model was built and tuned by changing the following parameters:
- rank: “80” used, which is the number of latent factors used for the model.
- regParam: “0.2” was used, which gave the best model when compared to other learning rates such as 0.1, 0.01, 1 etc.
- maxIter: “20” used, which is the number of iterations a model runs on the training dataset. 
- coldStartStrategy: “drop” - this was used to drop unknown ids absent from training dataset, because it wouldn’t be able to predict the ratings and also increase our cost, which are unsuitable for our model.
- nonnegative: “true” was set as we did not want our rating to go into the negative scale.
- userCol: “reviewer_id” - converted “reviewerID” to integer type suitable for our model.
- itemCol: “product_id” - converted “asin” to integer type suitable for our model.
- ratingCol: “rating” - converted “overall” to float data type suitable for our model.
- predictionCol: “prediction”, the rating prediction column.

- Evaluation of ALS: Train RMSE = 0.9987,  MAE = 0.7905



