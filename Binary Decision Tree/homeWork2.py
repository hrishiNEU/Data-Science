import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import *

##########################################
## Given D-Tree Hyper-parameters ranges
##########################################
max_depth_values = [3, 5]
min_samples_split_values = [5, 10]
min_samples_leaf_values = [3, 5]
min_impurity_decrease_values = [0.01, 0.001]
ccp_alpha_values = [0.001, 0.0001]

def input_data(file_name, label_name):
    data = pd.read_csv(file_name)
    labels = data.loc[:, data.columns == label_name]
    feats = data.loc[:, data.columns != label_name]
    return feats, labels

# Taking the input and specifying the Key column as Outcome
features, labels = input_data(file_name="diabetes.csv", label_name='Outcome')

# Splitting the data into train, test, and validation sets (72%, 20%, 8%)
# Adding the random state with the commonly used arbitrary integer to split the date correctly.
train_feat, temp_feat, train_label, temp_label = train_test_split(features, labels, test_size=0.28, random_state=42)
valid_feat, test_feat, valid_label, test_label = train_test_split(temp_feat, temp_label, test_size=0.2857, random_state=42)

best_accuracy = 0
best_params = {}

# Hyper-parameter tuning loop
for max_depth in max_depth_values:
    for min_samples_split in min_samples_split_values:
        for min_samples_leaf in min_samples_leaf_values:
            for min_impurity_decrease in min_impurity_decrease_values:
                for ccp_alpha in ccp_alpha_values:
                    # Create a decision tree model with current hyperparameters
                    model = tree.DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_impurity_decrease=min_impurity_decrease,
                        ccp_alpha=ccp_alpha
                    )

                    # Training the model on the training data using fit function
                    model.fit(train_feat, train_label)

                    # Validating the model on the validation data
                    valid_pred_label = model.predict(valid_feat)
                    accuracy = accuracy_score(valid_pred_label, valid_label)

                    # Check if this model is the best so far
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'min_impurity_decrease': min_impurity_decrease,
                            'ccp_alpha': ccp_alpha
                        }

# Train a model using the best hyperparameters found
best_model = tree.DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_impurity_decrease=best_params['min_impurity_decrease'],
    ccp_alpha=best_params['ccp_alpha']
)

# Train the model on the entire training data
best_model.fit(train_feat, train_label)

# Calculate accuracy on the test data
test_pred_label = best_model.predict(test_feat)
test_accuracy = accuracy_score(test_pred_label, test_label)
print('Test Accuracy =', test_accuracy)

#Decision Tree will come here
plt.figure(figsize=(10, 10))
tree.plot_tree(
    best_model,
    feature_names=train_feat.columns,
    class_names=['Negative', 'Positive'],  # Assuming 'Outcome' is 0 or 1
    filled=True
)
plt.show()

# Confusion Matrix will come here
conf_matrix = confusion_matrix(test_label, test_pred_label)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Negative', 'Positive']).plot()
plt.show()

# To print the tree's text
tree_rules = tree.export_text(best_model, feature_names=train_feat.columns.tolist())

# Metrics for the evaluation
Accuracy = accuracy_score(test_label, test_pred_label)
Sensitivity = recall_score(test_label, test_pred_label)
Specificity = recall_score(test_label, test_pred_label, pos_label=0)

print("Accuracy:", Accuracy, "Sensitivity:", Sensitivity, "Specificity:", Specificity)