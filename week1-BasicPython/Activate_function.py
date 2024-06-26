import math
import numpy as np

def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

def calc_sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calc_relu(x):
    return np.maximum(0, x)

def calc_elu(x, alpha=0.01):
    return x if x > 0 else alpha * (math.exp(x) - 1)

def calc_activation_function(x,activation_function:str) -> float:
    if is_number(x):
        x = float(x)
        if activation_function == "sigmoid":
            print("sigmoid: f({0})=".format(x), calc_sigmoid(x))
            return calc_sigmoid(x)
        elif activation_function == "relu":
            print("relu: f({0})=".format(x), calc_relu(x))
            return calc_relu(x)
        elif activation_function == "elu":
            print("elu: f({0})=".format(x), calc_elu(x, 0.01))
            return calc_elu(x)
        else:
            print(activation_function, " is not supported")
            
    else:
        print("x must be a number")
        
if __name__ == "__main__":
    print((calc_activation_function(x=1.5, activation_function='sigmoid')))
    print((calc_activation_function(x="abc", activation_function='sigmoid')))
    print((calc_activation_function(x=1.5, activation_function='belu')))
    #Câu 2:
    print(is_number (1) )
    print(is_number ("n"))
    #Câu 4:
    assert np.isclose(round(calc_sigmoid(3), 2), 0.95, rtol=1e-09, atol=1e-09)
    print(round(calc_sigmoid(2), 2))
    #Câu 5:
    assert round(calc_elu(1)) ==1
    print(round(calc_elu(-1), 2))
    #Câu 6:
    print(round(calc_activation_function(x = 3,activation_function='sigmoid'),2))
    exit(0)



