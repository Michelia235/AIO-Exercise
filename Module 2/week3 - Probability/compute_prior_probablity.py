import numpy as np

def create_train_data():
    data = [
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes']
    ]
    return np.array(data)

def compute_prior_probablity(train_data):
    y_unique = ['No', 'Yes']
    prior_probability = np.zeros(len(y_unique))
    total = train_data.shape[0]
    for i, y in enumerate(y_unique):
        prior_probability[i] = np.sum(train_data[:, -1] == y) / total
    return prior_probability

train_data = create_train_data()
print(train_data)
prior_probability = compute_prior_probablity(train_data)
print("P(play tennis = No):", prior_probability[0])
print("P(play tennis = Yes):", prior_probability[1])
