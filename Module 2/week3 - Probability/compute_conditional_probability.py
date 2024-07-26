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

def compute_conditional_probability(train_data):
    y_unique = ['No', 'Yes']
    conditional_probability = []
    list_x_name = []

    for i in range(train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)
        x_conditional_probability = np.zeros((len(y_unique), len(x_unique)))
        
        for j, y in enumerate(y_unique):
            y_count = np.sum(train_data[:, -1] == y)
            for k, x in enumerate(x_unique):
                x_count = np.sum((train_data[:, i] == x) & (train_data[:, -1] == y))
                x_conditional_probability[j, k] = x_count / y_count
        conditional_probability.append(x_conditional_probability)

    return conditional_probability, list_x_name

train_data = create_train_data()

_, list_x_name = compute_conditional_probability(train_data)
for i, feature_name in enumerate(list_x_name, start=1):
    print(f"x{i} = {feature_name}")

print(train_data.shape[1])






