import numpy as np
import matplotlib.pyplot as plt
import random

def get_column(data, index):
    # Lấy cột dữ liệu theo chỉ số
    result = [row[index] for row in data]
    return result
def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()

    # Get tv (index=0)
    tv_data = get_column(data, 0)

    # Get radio (index=1)
    radio_data = get_column(data, 1)

    # Get newspaper (index=2)
    newspaper_data = get_column(data, 2)

    # Get sales (index=3)
    sales_data = get_column(data, 3)

    # Build X input and y output for training
    X = [[1, x1, x2, x3] for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]
    y = sales_data
    return X, y

def initialize_params():
    b = 0
    w1 = random.gauss(mu=0.0, sigma=0.01)
    w2 = random.gauss(mu=0.0, sigma=0.01)
    w3 = random.gauss(mu=0.0, sigma=0.01)
    
    return [0 , -0.01268850433497871 , 0.004752496982185252 , 0.0073796171538643845]
    # return [b, w1, w2, w3]

def predict(X, w):
    return sum([x * wi for x, wi in zip(X, w)])

def compute_loss(y_hat, y):
    return (y_hat - y) ** 2

def compute_gradient_w(X, y, y_hat):
    return [2 * xi * (y_hat - y) for xi in X]

def update_weight(weights, dl_dweights, lr):
    return [wi - lr * dl_dwi for wi, dl_dwi in zip(weights, dl_dweights)]

def implement_linear_regression(X, y, epoch_max=50, lr=1e-5):
    losses = []
    weights = initialize_params()
    N = len(y)
    for epoch in range(epoch_max):
        print('Epoch ', epoch)
        for i in range(N):
            X_i = X[i]
            y_i = y[i]

            y_pred = predict(X_i, weights)
            loss = compute_loss(y_pred, y_i)

            dl_dw = compute_gradient_w(X_i, y_i, y_pred)
            weights = update_weight(weights, dl_dw, lr)

            # Logging
            losses.append(loss)

    return weights, losses


X, y = prepare_data(r'C:\Users\Administrator\Desktop\AIO-Exercise\Module 4\week1 - Linear Regression\advertising.csv')
W, L = implement_linear_regression(X, y)
print(L[9999])