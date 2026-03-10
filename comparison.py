"""
Model: Decision Tree
The decision tree model approximates the function it is learning by creating
many constant segments. It does so through a series of if, then decisions which
determine which constant the function is approximated to. I chose this model 
because I found it interesting and unique. I like that you can visually follow
the tree unlike a deep neural network where you have hidden layers that you 
can't look into. 
"""

import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from binary_classification import load_data


def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train.numpy(), y_train.numpy())
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Make predictions
    train_pred = torch.tensor(model.predict(X_train.numpy()), dtype=torch.float32)
    test_pred = torch.tensor(model.predict(X_test.numpy()), dtype=torch.float32)
    
    # Compute accuracies using torch
    train_acc = (y_train == train_pred).float().mean().item()
    test_acc = (y_test == test_pred).float().mean().item()
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test.numpy(), test_pred.numpy()))
    print("Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test.numpy(), test_pred.numpy()))
    
    return train_acc, test_acc


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {X_train.shape[1]}\n")
    
    # Train decision tree with default depth
    print("Training Decision Tree (no depth limit)...")
    dt_model = train_decision_tree(X_train, y_train, max_depth=None)
    print("Training complete!\n")
    
    evaluate_model(dt_model, X_train, y_train, X_test, y_test)
    
    # Train with limited depth for comparison
    print("\n" + "="*50)
    print("Training Decision Tree (max_depth=5)...")
    dt_model_limited = train_decision_tree(X_train, y_train, max_depth=5)
    print("Training complete!\n")
    
    evaluate_model(dt_model_limited, X_train, y_train, X_test, y_test)
