import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Load dataset
data = fetch_olivetti_faces()

# Dataset info
print("Data Shape:", data.data.shape)
print("Target Shape:", data.target.shape)
print("There are {} unique persons in the dataset".format(len(np.unique(data.target))))
print("Size of each image is {} x {}".format(data.images.shape[1], data.images.shape[2]))

# Function to display faces
def print_faces(images, target, top_n):
    top_n = min(top_n, len(images))
    grid_size = int(np.ceil(np.sqrt(top_n)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.ravel()):
        if i < top_n:
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f"Person: {target[i]}", fontsize=8)
        else:
            ax.axis('off')

    plt.show()

# Show some faces
print_faces(data.images, data.target, 25)

# Display one image per person
def display_unique_faces(images):
    fig = plt.figure(figsize=(12, 6))
    columns, rows = 10, 4

    for i in range(1, columns * rows + 1):
        img_index = 10 * i - 1
        if img_index < images.shape[0]:
            img = images[img_index]
            ax = fig.add_subplot(rows, columns, i)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Person {i}", fontsize=8)
            ax.axis('off')

    plt.suptitle("40 Distinct Persons", fontsize=16)
    plt.show()

display_unique_faces(data.images)

# Train-test split
from sklearn.model_selection import train_test_split

X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("x_train:", x_train.shape)
print("x_test:", x_test.shape)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

print("\nGaussian Naive Bayes Results")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy, "%")

# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(x_train, y_train)

y_pred_m = mnb.predict(x_test)

accuracy_m = round(accuracy_score(y_test, y_pred_m) * 100, 2)

print("\nMultinomial Naive Bayes Results")
print("Accuracy:", accuracy_m, "%")

# Misclassified images (Gaussian NB)
misclassified_idx = np.where(y_pred != y_test)[0]
num_misclassified = len(misclassified_idx)

print("\nNumber of misclassified images:", num_misclassified)
print("Total test images:", len(y_test))
print("Accuracy (manual):", round((1 - num_misclassified / len(y_test)) * 100, 2), "%")

# Show misclassified images
n_show = min(5, num_misclassified)

plt.figure(figsize=(10, 4))
for i in range(n_show):
    idx = misclassified_idx[i]
    plt.subplot(1, n_show, i + 1)
    plt.imshow(x_test[idx].reshape(64, 64), cmap='gray')
    plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
    plt.axis('off')

plt.show()