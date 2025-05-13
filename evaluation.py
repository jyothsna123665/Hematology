from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Example true and predicted blood groups
true_labels = ['A+', 'B+', 'O-', 'AB+', 'A-', 'O+', 'B-', 'A+']
predicted_labels = ['A+', 'B+', 'O+', 'AB+', 'A-', 'O+', 'O-', 'A+']

# Convert string labels to numeric format
encoder = LabelEncoder()
true_encoded = encoder.fit_transform(true_labels)
pred_encoded = encoder.transform(predicted_labels)

# Calculate metrics
accuracy = accuracy_score(true_encoded, pred_encoded)
precision = precision_score(true_encoded, pred_encoded, average='macro')
recall = recall_score(true_encoded, pred_encoded, average='macro')
f1 = f1_score(true_encoded, pred_encoded, average='macro')

# Print results
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
