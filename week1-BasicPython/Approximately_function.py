def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def approx_sin(x, n):
    return sum([(-1)** i * x ** (2 * i + 1) / factorial(2 * i + 1) for i in range(n)])

def approx_cos(x, n):
    return  sum([(-1)** i * x ** (2 * i) / factorial(2 * i) for i in range(n)])

def approx_sinh(x, n):
    return sum([x ** (2 * i + 1) / factorial(2 * i + 1) for i in range(n)])

def approx_cosh(x, n):
    return sum([x ** (2 * i) / factorial(2 * i) for i in range(n)])

x = 3.14
n = 10
print(approx_sin(x, n))
print(approx_cos(x, n))
print(approx_sinh(x, n))
print(approx_cosh(x, n))

#Câu 9:
assert round(approx_cos(x=1,n=10) , 2) == 0.54
print(round(approx_cos(x,n) , 2) )

#Câu 10:
assert round (approx_sin(x =1, n =10), 4) ==0.8415
print (round(approx_sin(x,n ) , 4) )

#Câu 11:
assert round(approx_sinh(x =1, n =10), 2) ==1.18
print(round(approx_sinh(x, n),2))

#Câu 12:
assert round(approx_cosh(x =1, n =10), 2) ==1.54
print(round(approx_cosh(x, n),2))
