import pandas as pd
import numpy as np
import math
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import combinations

train_1 = pd.read_csv('train.csv') 
train_1 = train_1.sample(frac=1) # xáo trộn dữ liệu

# Đọc tập kiểm tra
test_1 = pd.read_csv('test.csv')

# Danh sách 23 đặc trưng
all_features = ['Gender', '10percentage', '12percentage', 'CollegeTier', 'Degree', 'collegeGPA', 'CollegeCityTier',
                'English', 'Logical', 'Quant', 'Domain', 'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience',
                'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg', 'conscientiousness', 'agreeableness',
                'extraversion', 'nueroticism', 'openess_to_experience']

def calculate_mae_for_feature_combination(data, feature_combination):
    # Chuyển tuple thành danh sách
    feature_list = list(feature_combination)
    
    # Tạo DataFrame chứa các đặc trưng cần kiểm tra và cột 'Salary'
    train_data = data[feature_list + ['Salary']]
    
    # Xây dựng mô hình hồi quy tuyến tính
    reg = LinearRegression().fit(train_data[feature_list], train_data['Salary'])
    
    # Sử dụng k-fold Cross Validation để tính Mean Absolute Error (MAE)
    mae_scores = -cross_val_score(reg, train_data[feature_list], train_data['Salary'], cv=5, scoring='neg_mean_absolute_error')
    mean_mae = np.mean(mae_scores)
    
    return mean_mae

def create_linear_formula(coefficients, intercept, feature_names):
    formula_parts = [f'{intercept:.2f}']  # Thêm hệ số chặn vào chuỗi công thức

    for coef, feature_name in zip(coefficients, feature_names):
        if coef >= 0:
            formula_parts.append(f"+ {coef:.2f} * {feature_name}")
        else:
            formula_parts.append(f"- {abs(coef):.2f} * {feature_name}")

    linear_formula = " ".join(formula_parts)
    return linear_formula

def find_best_feature_combination(data, features, num_features):
    best_mae = float('inf')  # Khởi tạo MAE tốt nhất ban đầu là vô cùng lớn
    best_combination = None   # Lưu lại tổ hợp tốt nhất
    
    # Tạo tất cả các tổ hợp có thể của num_features đặc trưng từ danh sách
    feature_combinations = combinations(features, num_features)
    
    for feature_combination in feature_combinations:
        # Tính MAE cho tổ hợp đặc trưng hiện tại
        current_mae = calculate_mae_for_feature_combination(data, feature_combination)
        
        # Kiểm tra xem tổ hợp hiện tại có tốt hơn tổ hợp tốt nhất hiện tại hay không
        if current_mae < best_mae:
            best_mae = current_mae
            best_combination = feature_combination
    
    # Tạo mô hình hồi quy tốt nhất với tổ hợp tốt nhất
    best_feature_names = list(best_combination)
    best_model = LinearRegression().fit(data[best_feature_names], data['Salary'])
    
    # Dự đoán giá trị trên tập kiểm tra
    predictions = best_model.predict(test_1[best_feature_names])

    # Tính Mean Absolute Error trên tập kiểm tra
    mae_test = mean_absolute_error(test_1['Salary'], predictions)
    print(f"Mean Absolute Error on test set: {mae_test:.2f}")

    
    # Tạo hàm y(Salary) từ mô hình tốt nhất
    linear_formula = create_linear_formula(best_model.coef_, best_model.intercept_, best_feature_names)
    
    return best_combination, mae_test, linear_formula

def model(num_features):
    best_combination, best_mae, linear_formula = find_best_feature_combination(train_1, all_features, num_features)
    print(f"Best feature combination: {best_combination}")
    print(f"Best Mean Absolute Error: {best_mae:.2f}")
    print("Linear formula:", linear_formula)
    return best_combination, best_mae, linear_formula


def main():
    mae_scores = []
    for i in range(2,5):
        start_time = time.time()
        mae=0
        if i == 2:
            _,mae,_ = model(2)
        elif i == 3:
            _,mae,_ = model(3)
        elif i == 4:
            _,mae,_ = model(4)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Thời gian chạy: {execution_time:.2f} giây\n\n")
        mae_scores.append(mae)
    
    best_combination, mae_test, linear_formula = model(np.argmin(mae_scores)+2)
    print(f"Best feature combination: {best_combination}")
    print(f"Best Mean Absolute Error: {mae_test:.2f}")
    print("Linear formula:", linear_formula)
    

if __name__ == "__main__":
    main()
