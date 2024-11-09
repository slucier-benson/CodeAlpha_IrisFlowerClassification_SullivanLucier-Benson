# Iris Flower Classification Project

## Project Objectives
This project aims to classify iris flowers into three species: Setosa, Versicolor, and Virginica, using machine learning techniques. By analyzing the measurements of sepals and petals, we build a model to accurately predict the species of an iris flower based on its characteristics.

## Data Source
The dataset used in this project is the well-known **Iris dataset** provided by `scikit-learn`. It contains 150 samples of iris flowers with four features: sepal length, sepal width, petal length, and petal width.

## Model Choice
We used the **K-Nearest Neighbors (KNN)** classifier with `k=3` as the primary model for this classification task. The KNN model is simple yet effective for small datasets and provides a baseline for classification accuracy.

## Results
The model achieved an accuracy of approximately **100.00%** on the test set. Here is a detailed classification report:


Additional visualizations, such as pair plots and box plots, were generated to explore feature distributions and relationships.

## Instructions to Run
1. Ensure `scikit-learn`, `pandas`, and `matplotlib` are installed.
2. Run the Python script `iris_classification.py` to load data, visualize, train, and evaluate the model.
