import matplotlib.pyplot as plt

# Data for units 16, 32, 48, and 64
data = {

    16: {
        'accuracy': [0.8860759735107422, 0.8860759735107422, 0.8607594966888428, 0.8860759735107422, 0.8734177350997925],
        'precision': [0.8409090909090909, 0.8409090909090909, 0.85, 0.8947368421052632, 0.8536585365853658],
        'recall': [0.9487179487179487, 0.9487179487179487, 0.8717948717948718, 0.8717948717948718, 0.8974358974358975],
        'f1_score': [0.891566265060241, 0.891566265060241, 0.8607594936708861, 0.8831168831168831, 0.875]
    },

    32: {
        'accuracy': [0.8987341523170471, 0.8734177350997925, 0.8734177350997925, 0.8987341523170471, 0.8987341523170471],
        'precision': [0.8780487804878049, 0.8717948717948718, 0.8222222222222222, 0.8780487804878049, 0.8974358974358975],
        'recall': [0.9230769230769231, 0.8717948717948718, 0.9487179487179487, 0.9230769230769231, 0.8974358974358975],
        'f1_score': [0.9, 0.8717948717948718, 0.8809523809523809, 0.9, 0.8974358974358975]
    },

    48: {
        'accuracy': [0.8734177350997925, 0.8734177350997925, 0.8734177350997925, 0.8987341523170471, 0.8607594966888428],
        'precision': [0.8536585365853658, 0.9142857142857143, 0.8717948717948718, 0.8780487804878049, 0.8888888888888888],
        'recall': [0.8974358974358975, 0.8205128205128205, 0.8717948717948718, 0.9230769230769231, 0.8205128205128205],
        'f1_score': [0.875, 0.8648648648648648, 0.8717948717948718, 0.9, 0.8533333333333333]
    },

    64: {
        'accuracy': [0.8860759735107422, 0.8987341523170471, 0.8860759735107422, 0.8860759735107422, 0.8987341523170471],
        'precision': [0.8947368421052632, 0.8780487804878049, 0.8409090909090909, 0.8571428571428571, 0.8780487804878049],
        'recall': [0.8717948717948718, 0.9230769230769231, 0.9487179487179487, 0.9230769230769231, 0.9230769230769231],
        'f1_score': [0.8831168831168831, 0.9, 0.891566265060241, 0.888888888888889, 0.9]
    }
}

# Calculate averages
averages = {
    'units': [16, 32, 48, 64],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}

for unit, metrics in data.items():
    averages['accuracy'].append((sum(metrics['accuracy']) / len(metrics['accuracy'])) * 100)
    averages['precision'].append((sum(metrics['precision']) / len(metrics['precision'])) * 100)
    averages['recall'].append((sum(metrics['recall']) / len(metrics['recall'])) * 100)
    averages['f1_score'].append((sum(metrics['f1_score']) / len(metrics['f1_score'])) * 100)

# Extract data from averages
units = averages['units']
accuracy = averages['accuracy']
recall = averages['recall']
precision = averages['precision']
f1_score = averages['f1_score']

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(units, accuracy, marker='o', label='Accuracy', color='b', linestyle='-')
plt.plot(units, recall, marker='o', label='Recall', color='g', linestyle='--')
plt.plot(units, precision, marker='s', label='Precision', color='r', linestyle='-.')
plt.plot(units, f1_score, marker='^', label='F1 Score', color='m', linestyle=':')

# Add title and labels
plt.title('Model Performance Metrics Over Different Attention Units')
plt.xlabel('Units')
plt.ylabel('Score (%)')

# Add a legend
plt.legend()

# Add grid
plt.grid(True)

# Show the plot
plt.show()
