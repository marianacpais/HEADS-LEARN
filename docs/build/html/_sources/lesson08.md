# Lesson 08 - Unsupervised Machine Learning - Data Preprocessing and Anomaly Detection

## Summary

- CRISP-DM
- There are basically 3 approaches to sampling:
  - Simple random sampling (with or without replacement)
  - Stratified sampling (e.g., relative to class, maintaining the same number of objects for each class, or maintaining the same proportion of objects from the original set)
  - Progressive sampling (progressively increasing the sample size as long as the accuracy rate continues to improve; usually this approach provides a good estimate for the sample size)
- Anomaly detection approaches
  - Type 1 – Determine the outliers with no prior knowledge of the data; essentially a learning approach analogous to unsupervised clustering
  - Type 2 – Model both normality and abnormality; approach analogous to supervised classification and requires pre-labelled data, tagged as normal or abnormal
  - Type 3 – Model only normality or in a very few cases model abnormality; analogous to a semi-supervised recognition or detection task
- Types of missing data: A) Missing completely at random; B) Missing at random; C) Missing not at random
