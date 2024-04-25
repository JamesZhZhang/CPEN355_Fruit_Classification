# The holy trinity of data analysis
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Used for data importing
import os
import cv2

# Sklearn imports for model building
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# For hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

# Extra imports for plotting decision region
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_decision_regions

# Globals for importing
TRAIN_PATH = "fruits-360/Training/"
TEST_PATH = "fruits-360/Test/"
FRUITS_TO_CLASSIFY = ["Apple Red 1", "Apple Red 2"]
TRAIN_SAMPLE_SIZE = 200
TEST_SAMPLE_SIZE = 86   # Approximate 70/30 split

# Other globals
N_FOLDS = 5

def load_images(path:str = TRAIN_PATH, sample_size:int = TRAIN_SAMPLE_SIZE, fruits_to_classify:list = FRUITS_TO_CLASSIFY) -> tuple[np.array, np.array]:

    images = []
    labels = []
    for fruit in os.listdir(path):
        label = fruit.split('/')[-1]
        if label in fruits_to_classify:
            files = os.listdir(path + label)
            num_files = len(files)
            indices = np.linspace(0, num_files - 1, sample_size, dtype=int)
            for index in indices:
                image_path = os.path.join(path + fruit, files[index])
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert imread's BGR reading to RGB for image reconstruction
                image = image.flatten()  # Flatten the image into a 1D array
                images.append(image)
                labels.append(label)

    value_map = {"Apple Red 1": 1, "Apple Red 2": 2}
    labels_mapped = pd.Series(labels).map(value_map).values

    images = np.array(images)
    labels = np.array(labels_mapped)
    return images, labels

def plot_images(X, num_images=20):
    num_rows = int(np.ceil(np.sqrt(num_images)))
    num_cols = int(np.ceil(num_images / num_rows))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_images):
        image = X[i].reshape(100, 100, 3)  # Reshape the flattened image vector to a 2D array
        image = image[:, :, ::-1]  # Convert color format from BGR to RGB
        axes[i].imshow(image)
        axes[i].axis('off')
    
    for i in range(num_images, num_rows * num_cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def fit_PCA(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

def plot_reconstructed(X_train_pca:np.ndarray = None, pca:PCA = None, X_train:np.ndarray = None, index:int = 0):
    reconstructed_image = pca.inverse_transform(X_train_pca[index].reshape(1, -1))

    # Reshape the original and reconstructed images to their original shape
    image_shape = (100, 100, 3)  # Assuming the original image shape is (100, 100, 3)
    original_image = X_train[index].reshape(image_shape)
    reconstructed_image = reconstructed_image.reshape(image_shape)

    # Plot the original and reconstructed images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(original_image.astype(np.uint8))
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(reconstructed_image.astype(np.uint8))
    ax2.set_title('Reconstructed Image (PCA)')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def train_and_evaluate_svc(X_train, y_train, X_test, y_test):
    Cs = [0.00000000001, 0.0000001, 0.00001, 0.001, 0.01, 1, 100, 1000, 100000, 1000000000] # TODO: fill in the hyper-parameter candidates
    accuracies = []
    for c in Cs:        
        # Create a LinearSVC classifier with C=1
        svc = LinearSVC(random_state=2, C=c, dual=False, max_iter=10000)

        # Train the classifier on the training data
        svc.fit(X_train, y_train)

        accuracy = svc.score(X_test, y_test)

        accuracies.append(accuracy)

    return svc, accuracies

def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    n_trees = [1, 5, 10, 50, 100, 500, 1000]
    accuracies = []
    for n in n_trees:
        rf = RandomForestClassifier(n_estimators=n, random_state=2)
        rf.fit(X_train, y_train)
        accuracy = rf.score(X_test, y_test)
        accuracies.append(accuracy)
    return rf, accuracies

def train_and_evaluate_knn(X_train, y_train, X_test, y_test):
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        accuracies.append(accuracy)
    return knn, accuracies

def plot_scatter(X, y):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'purple']  # Adjust colors based on the number of unique labels
    for label, color in zip(unique_labels, colors):
        mask = (y == label)
        plt.scatter(X[mask, 0], X[mask, 1], c=color, label=label, alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('Scatter Plot of PCA-transformed Features')
    plt.show()

def plot_decision_boundary(X_train, y_train, X_test, y_test):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    X_train_pca, pca = fit_PCA(X_train, n_components=2)  # Reduce dimensionality to 2 for visualization
    X_test_pca = pca.transform(X_test)

    svc, accuracy = train_and_evaluate_svc(X_train_pca, y_train_encoded, X_test_pca, y_test_encoded)
    print(f"LinearSVC accuracy: {accuracy:.4f}")

    # Plot decision regions
    plot_decision_regions(X_train_pca, y_train_encoded, clf=svc, legend=2)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('Decision Regions')
    plt.show()

def main():
    X_train, y_train = load_images()
    X_test, y_test = load_images(path=TEST_PATH, sample_size=TEST_SAMPLE_SIZE)
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    # Reduce dimensionality by fitting a PCA and transforming X data
    X_train_pca, pca = fit_PCA(X_train)
    X_test_pca = pca.transform(X_test)
    #plot_reconstructed(X_train_pca, pca, X_train)
    #plot_scatter(X_train_pca, y_train) # Plot the separation of features after PCA

    print(X_train_pca.shape)

    # Fit linear kernel SVM and score accuracy on testing set
    svc, accuracies = train_and_evaluate_svc(X_train_pca, y_train, X_test_pca, y_test)
    print(f"LinearSVC accuracies: {accuracies}")

    plot_decision_regions(X_train_pca, y_train, clf=svc, legend=2)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('Decision Regions')
    plt.show()

    # Fit random forest and score accuracy on testing set
    rf, accuracies = train_and_evaluate_rf(X_train_pca, y_train, X_test_pca, y_test)
    print(f"RandomForest accuracies: {accuracies}")

    # Fit KNN and score accuracy on testing set
    knn, accuracies = train_and_evaluate_knn(X_train_pca, y_train, X_test_pca, y_test)
    print(f"KNN accuracies: {accuracies}")

    # Perform grid search to optimize model hyperparameters
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=2)

    space_lsvc = {'C' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    space_rf = {'n_estimators' : [1, 10, 50, 100, 500, 1000]}
    space_knn = {'n_neighbors' : [1, 3, 5, 7, 9, 11, 13, 15]}

    print("Creating grid search objects...")
    grid_lsvc = GridSearchCV(estimator = svc, param_grid = space_lsvc, cv=cv, scoring='accuracy')
    grid_rf = GridSearchCV(estimator = rf, param_grid = space_rf, cv=cv, scoring='accuracy')
    grid_knn = GridSearchCV(estimator = knn, param_grid = space_knn, cv=cv, scoring='accuracy')

    # Fit the grid search objects
    print("Fitting grid search objects...")
    grid_lsvc.fit(X_train_pca, y_train)
    grid_rf.fit(X_train_pca, y_train)
    grid_knn.fit(X_train_pca, y_train)

    # Create the best models and best predictions
    best_lsvc = grid_lsvc.best_estimator_
    lsvc_yhat = best_lsvc.predict(X_test_pca)
    best_rf = grid_rf.best_estimator_
    rf_yhat = best_rf.predict(X_test_pca)
    best_knn = grid_knn.best_estimator_
    knn_yhat = best_knn.predict(X_test_pca)

    print("Optimal C for linear kernel SVM: ", grid_lsvc.best_params_)
    print("Optimal hyperparameters for random forest: ", grid_rf.best_params_)
    print("Optimal hyperparameters for KNN: ", grid_knn.best_params_)

    print("LinearSVC accuracy: ", metrics.accuracy_score(y_test, lsvc_yhat))
    print("RandomForest accuracy: ", metrics.accuracy_score(y_test, rf_yhat))
    print("KNN accuracy: ", metrics.accuracy_score(y_test, knn_yhat))


if __name__ == "__main__":
    main()