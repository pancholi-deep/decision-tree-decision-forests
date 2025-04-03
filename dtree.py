import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(file_path):
    # Load data from the file
    data = np.loadtxt(file_path)
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels
    return X, y

def train_decision_tree(X_train, y_train, option):
    if option == "optimized":
        # Use a decision tree with optimal feature selection
        clf = DecisionTreeClassifier()
    elif option == "randomized" or option.startswith("forest"):
        # Use a decision tree with randomized feature selection
        clf = DecisionTreeClassifier(splitter="random")
    else:
        raise ValueError("Invalid option")

    if option.startswith("forest"):
        # If it's a forest, set the number of trees
        n_trees = int(option[len("forest"):])
        clf = RandomForestClassifier(n_estimators=n_trees)

    # Train the decision tree or forest
    clf.fit(X_train, y_train)
    return clf

def test_decision_tree(clf, X_test, y_test):
    # Test the decision tree
    y_pred = clf.predict(X_test)

    # Calculate accuracy for each test object
    accuracies = []
    for i in range(len(y_test)):
        pred_class = int(y_pred[i])
        true_class = int(y_test[i])
        accuracy = 1 if pred_class == true_class else 0
        accuracies.append(accuracy)

        print(f"Object Index = {i}, Result = {pred_class}, True Class = {true_class}, Accuracy = {accuracy}")

    # Calculate and print overall classification accuracy
    overall_accuracy = np.mean(accuracies)
    print(f"\nClassification Accuracy = {overall_accuracy}")

if __name__ == "__main__":
    import sys

    # Get command-line arguments
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    option = sys.argv[3]

    # Load training and test data
    X_train, y_train = load_data(training_file)
    X_test, y_test = load_data(test_file)

    # Train decision tree
    clf = train_decision_tree(X_train, y_train, option)

    # Test decision tree and print results
    test_decision_tree(clf, X_test, y_test)
