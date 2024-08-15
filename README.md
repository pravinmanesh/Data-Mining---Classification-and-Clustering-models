# Data-Mining---Classification-and-Clustering-models
This is an implementation of the simple types of Classification and Clustering models and comparing their accuracies
## Classification Overview

**Classification** is a supervised learning technique used in machine learning and statistics to assign a label or category to a given input data point. The goal of a classification algorithm is to learn from a labeled dataset (where the correct output labels are known) and make predictions about the labels of new, unseen data. Classification is widely used in various applications such as spam detection, sentiment analysis, medical diagnosis, and more.

### Key Concepts:
- **Labels**: The categories or classes into which data points are classified.
- **Features**: The attributes or properties of the data points used for classification.
- **Training Set**: A dataset used to train the model, consisting of input features and their corresponding labels.
- **Test Set**: A separate dataset used to evaluate the model's performance.

## Naive Bayes Classifier

The **Naive Bayes** classifier is a probabilistic model based on Bayes' theorem, with the assumption that the features are conditionally independent given the class label. Despite the "naive" assumption of independence, Naive Bayes often performs well in practice, especially in text classification tasks.

### How It Works:
- **Bayes' Theorem**: The classifier calculates the probability of each class given the input features and assigns the class with the highest probability. Bayes' theorem is given by:

  \[
  P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
  \]

  where:
  - \( P(C|X) \) is the posterior probability of class \( C \) given the features \( X \).
  - \( P(X|C) \) is the likelihood of features \( X \) given class \( C \).
  - \( P(C) \) is the prior probability of class \( C \).
  - \( P(X) \) is the probability of the features \( X \).

- **Independence Assumption**: The Naive Bayes classifier assumes that the features are independent of each other given the class. This simplifies the computation of the likelihood:

  \[
  P(X|C) = P(x_1|C) \cdot P(x_2|C) \cdot \ldots \cdot P(x_n|C)
  \]

### Types of Naive Bayes:
- **Gaussian Naive Bayes**: Assumes that the features follow a normal (Gaussian) distribution.
- **Multinomial Naive Bayes**: Typically used for discrete data like word counts in text classification.
- **Bernoulli Naive Bayes**: Used for binary/boolean features.

### Applications:
- Spam detection
- Text classification (e.g., sentiment analysis)
- Document categorization

## Decision Tree Classifier

A **Decision Tree** is a non-parametric, tree-like model used for both classification and regression tasks. It splits the data into subsets based on the values of the input features, creating a tree structure where each node represents a feature, each branch represents a decision rule, and each leaf node represents an output label.

### How It Works:
- **Tree Structure**: The decision tree is built by recursively splitting the dataset based on the feature that results in the highest information gain (or the lowest impurity) at each step.
- **Splitting Criteria**: Common criteria for splitting include:
  - **Gini Impurity**: Measures how often a randomly chosen element would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.
  - **Entropy (Information Gain)**: Measures the reduction in uncertainty or impurity when a dataset is split based on a feature.
  - **Chi-square**: Used in categorical features to check the significance of the split.

- **Stopping Criteria**: The tree grows until a stopping condition is met, such as a maximum depth, minimum number of samples in a node, or no further information gain.

### Advantages:
- **Interpretability**: The tree structure is easy to understand and visualize, making it interpretable even by non-experts.
- **Handling Non-linear Relationships**: Decision trees can capture non-linear relationships between features.

### Disadvantages:
- **Overfitting**: Decision trees can easily overfit the training data, especially if the tree is allowed to grow deep.
- **Instability**: Small changes in the data can result in a completely different tree structure.

### Applications:
- Credit scoring
- Medical diagnosis
- Fraud detection

## Clustering Overview

**Clustering** is an unsupervised learning technique used to group similar data points together into clusters. Unlike classification, clustering does not require labeled data. Instead, the goal is to find a structure in the data where similar items are grouped together, and dissimilar items are separated into different clusters. Clustering is widely used in various domains such as market segmentation, image segmentation, anomaly detection, and more.

### Key Concepts:
- **Clusters**: Groups of similar data points.
- **Centroid**: The center of a cluster, often used in algorithms like K-Means.
- **Distance Metrics**: Measures used to determine the similarity or dissimilarity between data points (e.g., Euclidean distance, Manhattan distance).

## K-Means Clustering

**K-Means** is a popular and simple clustering algorithm that aims to partition the data into \( k \) clusters, where each data point belongs to the cluster with the nearest centroid.

### How It Works:
1. **Initialization**: Select \( k \) initial centroids, either randomly or using specific methods like K-Means++.
2. **Assignment**: Assign each data point to the nearest centroid, forming \( k \) clusters.
3. **Update**: Recalculate the centroids of the clusters by averaging the points in each cluster.
4. **Iteration**: Repeat the assignment and update steps until the centroids no longer change significantly or a maximum number of iterations is reached.

### Advantages:
- **Simplicity**: K-Means is easy to implement and computationally efficient for large datasets.
- **Scalability**: Works well with large datasets.

### Disadvantages:
- **Fixed Number of Clusters**: The number of clusters \( k \) must be chosen in advance.
- **Sensitivity to Initialization**: Different initial centroids can lead to different results (local minima).
- **Assumes Spherical Clusters**: K-Means works best when clusters are roughly spherical and equally sized.

### Applications:
- Customer segmentation
- Image compression
- Document clustering

## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**DBSCAN** is a density-based clustering algorithm that groups together points that are closely packed together while marking points that lie alone in low-density regions as outliers.

### How It Works:
1. **Core Points**: Identify points that have at least a minimum number of neighbors (\( minPts \)) within a specified radius (\( \epsilon \)).
2. **Density-Connected Points**: Form clusters by connecting core points that are within \( \epsilon \) distance of each other.
3. **Outliers**: Points that do not belong to any cluster are labeled as outliers or noise.

### Advantages:
- **No Need to Specify \( k \)**: DBSCAN does not require the number of clusters to be specified in advance.
- **Robust to Outliers**: Effectively identifies and handles noise and outliers.
- **Flexible Cluster Shapes**: Can find clusters of arbitrary shape.

### Disadvantages:
- **Parameter Sensitivity**: The results depend on the choice of \( \epsilon \) and \( minPts \), which may require domain knowledge to set.
- **Difficulty with Varying Densities**: Struggles with data where clusters have varying densities.

### Applications:
- Geographic data analysis
- Anomaly detection
- Clustering spatial data

## Gaussian Mixture Model (GMM)

**Gaussian Mixture Model (GMM)** is a probabilistic model that assumes that the data is generated from a mixture of several Gaussian distributions with unknown parameters. Each cluster in the data is represented by a Gaussian distribution.

### How It Works:
1. **Initialization**: Initialize the parameters of the Gaussian distributions, such as the mean, covariance, and mixing coefficients.
2. **Expectation Step (E-Step)**: Calculate the probability that each data point belongs to each Gaussian distribution (responsibility).
3. **Maximization Step (M-Step)**: Update the parameters of the Gaussian distributions to maximize the likelihood of the data.
4. **Iteration**: Repeat the E-Step and M-Step until convergence.

### Advantages:
- **Soft Clustering**: Assigns probabilities to data points, indicating the likelihood of belonging to each cluster.
- **Flexibility**: Can model clusters with different shapes, sizes, and orientations.
- **Theoretical Foundation**: GMM has a solid statistical foundation and can be interpreted as a generative model.

### Disadvantages:
- **Complexity**: More complex and computationally expensive compared to K-Means.
- **Risk of Overfitting**: May require regularization to avoid overfitting, especially with small datasets.

### Applications:
- Image segmentation
- Speaker identification
- Anomaly detection

