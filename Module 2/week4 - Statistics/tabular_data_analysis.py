import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(
    r"C:\Users\Administrator\Desktop\AIO-Exercise\Module 2\week4 - Statistics\advertising.csv")
print(data)

# Question 5:


def correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))

    if denominator == 0:
        return 0

    return numerator / denominator


# Answer 5:
x = data['TV']
y = data['Radio']

corr_xy = correlation(x, y)
print(f"Correlation between TV and Sales : {round(corr_xy, 2)}")


# Question 6:
features = ['TV', 'Radio', 'Newspaper']

for feature_1 in features:
    for feature_2 in features:
        correlation_value = correlation(data[feature_1], data[feature_2])
        print(f"Correlation between {feature_1} and {
              feature_2}: {round(correlation_value, 2)}")

# Question 7:
x = data['Radio']
y = data['Newspaper']
result = np.corrcoef(x, y)
print(result)
# Expected output : [[1.    0.35410375]
#                    [0.35410375    1. ]]

# Question 8:
print(data.corr())
