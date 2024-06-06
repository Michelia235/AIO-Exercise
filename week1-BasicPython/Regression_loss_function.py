import random
import math
import numpy as np

def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

def exercise3():
    num = input("Enter number of samples: ")
    if not num.isnumeric():
        print("Number of samples must be a number")
        return
    loss_name = input("Enter loss name (MAE, MSE, RMSE): ")
    for i in range(int(num)):
        y_true = random.uniform(0, 10)
        y_pred = random.uniform(0, 10)
        print("lose name: ", loss_name)
        print("sample: {0}, pred: {1}, target: {2}".format(i, y_pred, y_true))
        if loss_name == "MAE":
            print("loss: ", mae(y_true, y_pred))
        elif loss_name == "MSE":
            print("loss: ", mse(y_true, y_pred))
        elif loss_name == "RMSE":
            print("loss: ", rmse(y_true, y_pred))
        else:
            print(loss_name, " is not supported")

#Câu 7:
print(mae(2,9))
#Câu 8:
print(mse(2,1))