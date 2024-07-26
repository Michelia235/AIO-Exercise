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

def get_index_from_value(feature_name, list_features):
    return np.nonzero(list_features == feature_name)[0][0]

train_data = create_train_data()

#Câu 16:
_, list_x_name = compute_conditional_probability(train_data)
outlook = list_x_name[0]
i1 = get_index_from_value("Overcast", outlook)
i2 = get_index_from_value("Rain", outlook)
i3 = get_index_from_value("Sunny", outlook)

print(i1, i2, i3)

#Câu 17:
train_data = create_train_data()
conditional_probability, list_x_name = compute_conditional_probability(train_data)
# Compute P(" Outlook "=" Sunny "| Play Tennis "=" Yes ")
x1 = get_index_from_value("Sunny", list_x_name[0])
print("P('Outlook' = 'Sunny' | 'Play Tennis' = 'Yes') =", np.round(conditional_probability[0][1, x1], 2))

#Câu 18:
# Compute P(" Outlook "=" Sunny "| Play Tennis "=" No ")
x1 = get_index_from_value("Sunny", list_x_name[0])
print("P('Outlook' = 'Sunny' | Play Tennis = 'No') =", np.round(conditional_probability[0][0, x1], 2))


