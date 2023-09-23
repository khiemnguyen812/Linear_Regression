import pandas as pd
import numpy as np
import math

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



# Phần code cho yêu cầu 1a
train_1 = pd.read_csv('train.csv') 
test_1 = pd.read_csv('test.csv')  


reg1 = LinearRegression().fit(train_1.iloc[:, :11], train_1.iloc[:, -1]) # Lấy 11 thuộc tính đầu tiên của train_1

# Hiển thị công thức hồi quy tuyến tính
coef = pd.DataFrame({'Feature': train_1.iloc[:, :11].columns, 'Coefficient': reg1.coef_})

# Lấy hệ số chặn (intercept)
intercept = reg1.intercept_

# Tạo chuỗi biểu diễn công thức hồi quy tuyến tính
formula_parts = [f'{intercept:.2f}']  # Thêm hệ số chặn vào chuỗi công thức

for index, row in coef.iterrows():
    if row['Coefficient'] >= 0:
        formula_parts.append(f"+ {row['Coefficient']:.2f} * {row['Feature']}")
    else:
        formula_parts.append(f"- {abs(row['Coefficient']):.2f} * {row['Feature']}")

print(coef)
linear_formula = " ".join(formula_parts)
print("y(Salary) = ", linear_formula)

mae = mean_absolute_error(test_1['Salary'], reg1.predict(test_1.iloc[:,:11]))
print("Mean Absolute Error (MAE):", mae)