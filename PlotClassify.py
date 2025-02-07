from sklearn.metrics import precision_score, recall_score, f1_score

# Ground truth labels (converted to numeric values for evaluation)
ground_truth = [4, 4, 4, 4, 4, 4, 1, 4, 4, 1, 4, 4, 4, 3, 1, 1, 1, 4, 4]

# Rule-Based Model Predictions
rule_based_pred = [3, 4, 4, 4, 4, 3, 1, 4, 4, 3, 4, 3, 4, 3, 1, 1, 1, 4, 4]

# GPT-Based Model Predictions
gpt_based_pred = [4, 4, 4, 3, 1, 1, 1, 0, 4, 4, 4, 3, 4, 3, 4, 2, 4, 4, 2]

# Compute Precision, Recall, and F1-score for Rule-Based Classifier
rule_based_precision = precision_score(ground_truth, rule_based_pred, average='macro')
rule_based_recall = recall_score(ground_truth, rule_based_pred, average='macro')
rule_based_f1 = f1_score(ground_truth, rule_based_pred, average='macro')

# Compute Precision, Recall, and F1-score for GPT-Based Classifier
gpt_based_precision = precision_score(ground_truth, gpt_based_pred, average='macro')
gpt_based_recall = recall_score(ground_truth, gpt_based_pred, average='macro')
gpt_based_f1 = f1_score(ground_truth, gpt_based_pred, average='macro')

# Store results
results = {
    "Rule-Based Classifier": [rule_based_precision, rule_based_recall, rule_based_f1],
    "GPT-Based Classifier": [gpt_based_precision, gpt_based_recall, gpt_based_f1]
}

import pandas as pd
import ace_tools as tools

# Create a DataFrame for display
df_results = pd.DataFrame(results, index=["Precision", "Recall", "F1 Score"])
tools.display_dataframe_to_user(name="Severity Classification Performance", dataframe=df_results)

# Return results for further analysis
df_results
