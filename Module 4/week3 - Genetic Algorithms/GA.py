import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import random
random.seed(0)

#Bài tập 1:
def load_data_from_file(fileName=r"C:\Users\Administrator\Desktop\AIO-Exercise\Module 4\week3 - Genetic Algorithms\advertising.csv"):
    # Tải dữ liệu từ file CSV
    data = np.genfromtxt(fileName, dtype=None, delimiter=',', skip_header=1)
    
    # Tách các đặc trưng (features) và doanh số (sales)
    features_X = data[:, :3]  # Lấy 3 cột đầu tiên làm đặc trưng
    sales_Y = data[:, 3]       # Lấy cột thứ 4 làm doanh số

    # Thêm cột chặn (intercept) vào ma trận đặc trưng
    intercept = np.ones((features_X.shape[0], 1))  # Tạo cột chặn có giá trị bằng 1
    features_X = np.concatenate((intercept, features_X), axis=1)  # Nối cột chặn với các đặc trưng

    return features_X, sales_Y  # Trả về các đặc trưng và doanh số



# Question 2:
features_X, _ = load_data_from_file()
print("Đáp án Quuestion 2:")
print(features_X[:5, :])

#Question 3:
_, sales_Y = load_data_from_file()
print("Đáp án Quuestion 3:")
print(sales_Y.shape)

#Bài tập 2:
def generate_random_value(bound=10):
    # Tạo giá trị ngẫu nhiên trong khoảng từ -bound/2 đến bound/2
    return (random.random() - 0.5) * bound

def create_individual(n=4, bound=10):
    # Tạo một cá thể với n giá trị ngẫu nhiên
    individual = [generate_random_value() for _ in range(n)]  # Tạo danh sách cá thể bằng cách gọi hàm generate_random_value
    return individual 

#Bài tập 3:
def compute_loss(individual):
    # Chuyển đổi cá thể thành mảng numpy
    theta = np.array(individual)
    
    # Tính toán giá trị dự đoán y_hat bằng cách nhân ma trận đặc trưng với theta
    y_hat = features_X.dot(theta)
    
    # Tính toán mất mát (loss) bằng cách tính bình phương độ lệch giữa y_hat và sales_Y, sau đó lấy trung bình
    loss = np.multiply((y_hat - sales_Y), (y_hat - sales_Y)).mean()
    return loss  

def compute_fitness(individual):
    # Tính toán mất mát cho cá thể
    loss = compute_loss(individual)
    
    # Tính toán độ thích nghi (fitness) từ mất mát
    fitness = 1 / (loss + 1)  # Thêm 1 để tránh chia cho 0
    return fitness 


#Question 4:
features_X, sales_Y = load_data_from_file()
individual = [4.09, 4.82, 3.10, 4.02]
fitness_score = compute_fitness(individual)
print("Đáp án Quuestion 4:")
print(fitness_score)

#Bài tập 4:
def crossover(individual1, individual2, crossover_rate=0.9):
    # Tạo bản sao của hai cá thể để không làm thay đổi cá thể gốc
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

    # Thực hiện phép lai (crossover) cho từng gen trong cá thể
    for i in range(len(individual1)):
        # Kiểm tra xem có nên trao đổi gen hay không dựa trên tỷ lệ crossover
        if random.random() < crossover_rate:
            individual1_new[i] = individual2[i]  # Trao đổi gen giữa hai cá thể
            individual2_new[i] = individual1[i]

    return individual1_new, individual2_new  

#Question 5:
individual1 = [4.09, 4.82, 3.10, 4.02]
individual2 = [3.44, 2.57, -0.79, -2.41]

individual1, individual2 = crossover(individual1, individual2, 2.0)
print("Đáp án Quuestion 5:")
print("individual1: ", individual1)
print("individual2: ", individual2)


#Bài tập 5:
def mutate(individual, mutation_rate=0.05):
    # Tạo bản sao của cá thể để không làm thay đổi cá thể gốc
    individual_m = individual.copy()

    # Thực hiện đột biến cho từng gen trong cá thể
    for i in range(len(individual)):
        # Kiểm tra xem có nên thực hiện đột biến hay không dựa trên tỷ lệ mutation
        if random.random() < mutation_rate:
            individual_m[i] = generate_random_value()  # Thay thế gen bằng giá trị ngẫu nhiên mới

    return individual_m  

#Question 6:
before_individual = [4.09, 4.82, 3.10, 4.02]
after_individual = mutate(individual, mutation_rate = 2.0)
print("Đáp án Quuestion 6:")
print(before_individual == after_individual)

#Bài tập 6:
def initializePopulation(m):
    # Khởi tạo quần thể với m cá thể ngẫu nhiên
    population = [create_individual() for _ in range(m)]
    return population 

#Bài tập 7:
def selection(sorted_old_population, m):
    # Chọn ngẫu nhiên hai chỉ số khác nhau từ quần thể đã được sắp xếp
    index1 = random.randint(0, m - 1)
    while True:
        index2 = random.randint(0, m - 1)
        if index2 != index1:  # Đảm bảo chỉ số thứ hai khác chỉ số thứ nhất
            break

    # Chọn cá thể tốt nhất giữa hai cá thể tại index1 và index2
    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]  # Cập nhật cá thể nếu index2 lớn hơn index1

    return individual_s  

#Bài tập 8:
def create_new_population(old_population, elitism=2, gen=1):
    # Lấy kích thước của quần thể cũ
    m = len(old_population)
    
    # Sắp xếp quần thể cũ dựa trên độ thích nghi
    sorted_population = sorted(old_population, key=compute_fitness)

    # In ra mất mát tốt nhất trong thế hệ hiện tại
    if gen % 1 == 0:
        print("Best loss:", compute_loss(sorted_population[m - 1]), "with chromosome:", sorted_population[m - 1])

    new_population = []
    # Tạo quần thể mới cho đến khi đạt kích thước mong muốn
    while len(new_population) < m - elitism:
        # Lựa chọn hai cá thể từ quần thể đã sắp xếp
        individual_s1 = selection(sorted_population, m)
        individual_s2 = selection(sorted_population, m)  # Có thể trùng lặp

        # Thực hiện phép lai (crossover)
        individual_t1, individual_t2 = crossover(individual_s1, individual_s2)

        # Thực hiện đột biến (mutation)
        individual_m1 = mutate(individual_t1)
        individual_m2 = mutate(individual_t2)

        # Thêm các cá thể mới vào quần thể mới
        new_population.append(individual_m1)
        new_population.append(individual_m2)

    # Thêm các cá thể tốt nhất từ quần thể cũ vào quần thể mới (elitism)
    for ind in sorted_population[m - elitism:]:
        new_population.append(ind.copy())

    return new_population, compute_loss(sorted_population[m - 1]) 


#Question 7:
individual1 = [4.09, 4.82, 3.10, 4.02]
individual2 = [3.44, 2.57, -0.79, -2.41]
old_population = [individual1, individual2]
print("Đáp án Quuestion 7:")
new_population, _ = create_new_population(old_population, elitism=2, gen=1)

#Bài tập 9:
def run_GA():
    # Đặt số thế hệ và kích thước quần thể
    n_generations = 100
    m = 600
    
    # Tải dữ liệu từ file
    features_X, sales_Y = load_data_from_file()
    
    # Khởi tạo quần thể ban đầu
    population = initializePopulation(m)
    losses_list = []  # Danh sách lưu trữ các mất mát qua từng thế hệ

    # Vòng lặp qua các thế hệ
    for i in range(n_generations):
        # Tạo quần thể mới và tính toán mất mát
        population, losses = create_new_population(population, 2, i)
        losses_list.append(losses)  # Lưu mất mát vào danh sách

    return losses_list, population

losses_list, population = run_GA()   
#Bài tập 10:
def visualize_loss(losses_list):
  plt.plot(losses_list, c='green')
  plt.xlabel('Generations')
  plt.ylabel('losses')
  plt.show()

#Bài tập 11:
def visualize_predict_gt():
    # Hiển thị giá trị thực tế và giá trị dự đoán
    sorted_population = sorted(population, key=compute_fitness)  # Sắp xếp quần thể theo độ thích nghi
    print(sorted_population[-1])  # In cá thể tốt nhất

    theta = np.array(sorted_population[-1])  # Chuyển cá thể tốt nhất thành mảng numpy

    estimated_prices = []  # Danh sách lưu trữ giá ước lượng
    # Tính toán giá ước lượng cho mỗi đặc trưng
    for feature in features_X:
        estimated_price = sum(c * x for x, c in zip(feature, theta))  # Tính giá ước lượng
        estimated_prices.append(estimated_price)  # Thêm giá ước lượng vào danh sách

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Samples') 
    plt.ylabel('Price')    
    plt.plot(sales_Y, c='green', label='Real Prices')  
    plt.plot(estimated_prices, c='blue', label='Estimated Prices')  
    plt.legend()  
    plt.show()

visualize_loss(losses_list)
visualize_predict_gt()
