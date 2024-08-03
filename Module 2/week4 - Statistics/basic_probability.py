#Question 1:
import numpy as np

def compute_mean(_):
  return np.sum(_) / len(_)


#Question 2:
def compute_median(_):
  size = len(_)
  X = np.sort(_)
  print(X)
  if (size % 2 == 0):
    return (1/2*(X[int(size/2)-1] \
                 + (X[int(size/2) + 1 - 1])))
  else:
    return X[int((size+1)/2)-1]

#Question 3:
def compute_std(_):
  mean = compute_mean(_)
  variance = 0
  for x in _:
    variance = variance + (x - mean)**2
  variance = variance / len(X)
  return np.sqrt(variance)

#Question 4:
def compute_correlation_cofficient(parameter_x, parameter_y):
  N = len(parameter_x)
  numerator = N * parameter_x.dot(parameter_y) - np.sum(parameter_x)*np.sum(parameter_y)
  denominator = np.sqrt(N*np.sum(np.square(parameter_x))-np.sum(parameter_x)**2) \
    * np.sqrt(N*np.sum(np.square(parameter_y))-np.sum(parameter_y)**2)

  return np.round(numerator / denominator,2)


#Answer 1:
X = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
print("Mean : ", compute_mean(X))

#Answer 2:
X = [1, 5, 4, 4, 9, 13]
print("Median: ", compute_median(X))

#Answer 3:
X = [ 171, 176, 155, 167, 169, 182]
print(np.round(compute_std(X),2))

#Answer 4:
X = np.asarray([-2, -5, -11, 6, 4, 15, 9])
Y = np.asarray([4, 25, 121, 36, 16, 225, 81])
print("Correlation: ", compute_correlation_cofficient(X,Y))