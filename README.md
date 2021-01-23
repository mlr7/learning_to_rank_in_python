# learning_to_rank_in_python
An exploration of learning-to-rank machine learning approaches in Python

![](img/triplepic.png)

### Introduction

Learning to Rank (LTR) algorithms are an interesting and useful sub-branch of machine learning. In this notebook we are going to explore several of these algorithms and have some fun with an open source ranking benchmark dataset.

### Dataset and Techniques

MSLR-WEB10K: Open source, 10,000 queries, relevance values from 0 to 4, schema: relevance, query id, feature vector

Random Baseline. Classification Baseline. RankNet. LambdaRank. LamdaMart. NDCG: Normalized Discounted Cumulative Gain

### Results and Discussion

![](img/conf_metrics_3.png)

![](img/feature_importance.png)

![](img/ltr_bars.png)

![](img/quantile_table.png)

![](img/density_plots.png)

We found that the ranking algorithms achieve a mean NCDG score higher than both the random model and the classification model, but it is closer than you might expect. LambdaRank also does slightly better than LambdMart, which also warrants further investigation. The next steps here will be to explore hyperparameter settings and expand the model training to a larger fraction of the dataset.