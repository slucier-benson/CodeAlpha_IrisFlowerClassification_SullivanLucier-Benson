# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Create pairwise scatter plots
pd.plotting.scatter_matrix(data, figsize=(10, 10), diagonal='kde')
plt.suptitle("Pairwise Scatter Plot of Iris Features")
plt.show()

# Plot each feature separately by species
features = data.columns[:-1]
for feature in features:
    data.boxplot(column=feature, by='species', figsize=(6, 4))
    plt.title(f'{feature} by Species')
    plt.suptitle("")  # Suppress the overall title for cleaner look
    plt.xlabel('Species')
    plt.ylabel(feature)
    plt.show()

from sklearn.model_selection import train_test_split
X = data.drop('species', axis=1)
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
# Initialize the model with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display a detailed classification report
print(classification_report(y_test, y_pred))



# Display the first few rows
print(data.head())
