import numpy as np
import matplotlib.pyplot as plt
import random


def get_column(data, index):
    # Lấy cột dữ liệu theo chỉ số
    result = [row[index] for row in data]
    return result


def prepare_data(file_name_dataset):
    # Đọc dữ liệu từ file
    data = np.genfromtxt(file_name_dataset, delimiter=',',
                         skip_header=1).tolist()

    # Lấy dữ liệu TV (index=0)
    tv_data = get_column(data, 0)

    # Lấy dữ liệu radio (index=1)
    radio_data = get_column(data, 1)

    # Lấy dữ liệu newspaper (index=2)
    newspaper_data = get_column(data, 2)

    # Lấy dữ liệu sales (index=3)
    sales_data = get_column(data, 3)

    # Xây dựng X đầu vào và y đầu ra cho việc huấn luyện
    X = [tv_data, radio_data, newspaper_data]
    y = sales_data
    return X, y


def implement_linear_regression(X_data, y_data, epoch_max=50, lr=1e-5):
    losses = []

    w1, w2, w3, b = initialize_params()

    N = len(y_data)
    for _ in range(epoch_max):
        for i in range(N):
            # lấy một mẫu
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]

            y = y_data[i]

            # tính toán đầu ra
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            # tính toán tổn thất
            loss = compute_loss_mse(y, y_hat)

            # tính gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            # cập nhật tham số
            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)

            # ghi lại
            losses.append(loss)
    return (w1, w2, w3, b, losses)


def implement_linear_regression_nsamples(X_data, y_data, epoch_max=50, lr=1e-5):
    losses = []

    w1, w2, w3, b = initialize_params()
    N = len(y_data)

    for _ in range(epoch_max):
        loss_total = 0.0
        dw1_total = 0.0
        dw2_total = 0.0
        dw3_total = 0.0
        db_total = 0.0

        for i in range(N):
            # Lấy một mẫu
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]
            y = y_data[i]

            # Tính toán đầu ra
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            # Tính toán tổn thất
            loss = compute_loss_mae(y, y_hat)

            # Cộng dồn tổn thất vào tổn thất tổng
            loss_total += loss

            # Tính toán gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            # Cộng dồn gradient w1, w2, w3, b
            dw1_total += dl_dw1
            dw2_total += dl_dw2
            dw3_total += dl_dw3
            db_total += dl_db

        # (sau khi xử lý N mẫu) - Cập nhật tham số
        w1 = update_weight_wi(w1, dw1_total / N, lr)
        w2 = update_weight_wi(w2, dw2_total / N, lr)
        w3 = update_weight_wi(w3, dw3_total / N, lr)
        b = update_weight_b(b, dl_db / N, lr)

        # Ghi log
        losses.append(loss_total / N)

    return (w1, w2, w3, b, losses)


def initialize_params():
    # w1 = random.gauss(mu=0.0, sigma=0.01)
    # w2 = random.gauss(mu=0.0, sigma=0.01)
    # w3 = random.gauss(mu=0.0, sigma=0.01)
    # b  = 0

    w1, w2, w3, b = (0.016992259082509283,
                     0.0070783670518262355, -0.002307860847821344, 0)
    return w1, w2, w3, b

# Tính toán đầu ra và tổn thất


def predict(x1, x2, x3, w1, w2, w3, b):
    # Tính toán đầu ra dự đoán
    result = w1 * x1 + w2 * x2 + w3 * x3 + b
    return result


def compute_loss_mse(y_hat, y):
    # Tính toán tổn thất MSE (Mean Squared Error)
    result = np.mean((y_hat - y) ** 2)
    return result


def compute_loss_mae(y_hat, y):
    # Tính toán tổn thất MAE (Mean Absolute Error)
    result = np.mean(np.abs(y_hat - y))
    return result

# Tính toán đạo hàm (gradient)


def compute_gradient_wi(xi, y, y_hat):
    # Tính toán gradient đối với trọng số wi
    dl_dwi = -2 * np.mean((y - y_hat) * xi)
    return dl_dwi


def compute_gradient_b(y, y_hat):
    # Tính toán gradient đối với bias b
    dl_db = -2 * np.mean(y - y_hat)
    return dl_db

# Cập nhật trọng số


def update_weight_wi(wi, dl_dwi, lr):
    # Cập nhật trọng số wi bằng gradient descent
    wi = wi - lr * dl_dwi
    return wi


def update_weight_b(b, dl_db, lr):
    # Cập nhật bias b bằng gradient descent
    b = b - lr * dl_db
    return b


X, y = prepare_data(
    r'C:\Users\Administrator\Desktop\AIO-Exercise\Module 4\week1 - Linear Regression\advertising.csv')

# Question 1:
list = [sum(X[0][:5]), sum(X[1][:5]), sum(X[2][:5]), sum(y[:5])]
print(f"Answer Question 2: \n{list}")

# Question 2:
y_p = predict(x1=1, x2=1, x3=1, w1=0, w2=0.5, w3=0, b=0.5)
print(f"Answer Question 2: \n{y_p}")

# Question 3:
l = compute_loss_mse(y_hat=1, y=0.5)
print(f"Answer Question 3: \n{l}")

# Question 4:
g_wi = compute_gradient_wi(xi=1.0, y=1.0, y_hat=0.5)
print(f"Answer Question 4: \n{g_wi}")

# Question 5:
g_b = compute_gradient_b(y=2.0, y_hat=0.5)
print(f"Answer Question 5: \n{g_b}")

# Question 6:
after_wi = update_weight_wi(wi=1.0, dl_dwi=-0.5, lr=1e-5)
print(f"Answer Question 6: \n{after_wi}")

# Question 7:
after_b = update_weight_b(b=0.5, dl_db=-1.0, lr=1e-5)
print(f"Answer Question 7: \n{after_b}")

# Question 8:
(w1, w2, w3, b, losses) = implement_linear_regression(X, y)
print(f"Answer Question 8: \n{w1, w2, w3}")

plt.plot(losses[:100])
plt.xlabel("#iteration")
plt.ylabel("Loss")
plt.show()

# Question 9:
tv = 19.2
radio = 35.9
newspaper = 51.3
sales = predict(tv, radio, newspaper, w1, w2, w3, b)
print(f"Answer Question 9: \npredicted sales is {sales}")

# Question 10:
l = compute_loss_mae(y_hat=1, y=0.5)
print(f"Answer Question 10: \n{l}")

# Question 11:
(w1, w2, w3, b, losses) = implement_linear_regression_nsamples(X,y,1000)
print(f"Answer Question 11: \n{w1, w2, w3}")

# Bonus
print(losses)
plt.plot(losses)
plt.xlabel("#epoch")
plt.ylabel("MSE Loss")
plt.show()
