import pandas as pd
import numpy as np
import math

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Phần code cho yêu cầu 1c
# Tìm ra đặc trưng ngoại ngữ, lô-gic, định lượng tốt nhất
# In ra các kết quả cross-validation như yêu cầu
train_1 = pd.read_csv('train.csv')
train_1 = train_1.sample(frac=1)
# Đọc tập kiểm tra
test_1 = pd.read_csv('test.csv')

numerical_features = ['English', 'Logical', 'Quant']

# Tạo DataFrame để lưu kết quả của k-fold Cross Validation cho mỗi đặc trưng
cv_results_c = pd.DataFrame(columns=['Feature', 'Mean_MAE'])

for feature in numerical_features:
    # Tạo DataFrame chứa đặc trưng cần kiểm tra và cột 'Salary'
    train_c = train_1[[feature, 'Salary']]
    
    # Xây dựng mô hình hồi quy tuyến tính
    reg_c = LinearRegression().fit(train_c[[feature]], train_c['Salary'])
    
    # Sử dụng k-fold Cross Validation để tính Mean Absolute Error (MAE)
    mae_scores = -cross_val_score(reg_c, train_c[[feature]], train_c['Salary'], cv=5, scoring='neg_mean_absolute_error')
    mean_mae = np.mean(mae_scores)
    
    # Lưu kết quả vào DataFrame
    cv_results_c = pd.concat([cv_results_c, pd.DataFrame({'Feature': [feature], 'Mean_MAE': [mean_mae]})], ignore_index=True)

print(cv_results_c)
# Chọn đặc trưng ngoại ngữ, lô-gic, định lượng tốt nhất
best_feature_c = cv_results_c.loc[cv_results_c['Mean_MAE'].idxmin()]['Feature']

print("Best numerical feature:", best_feature_c)


# Xây dựng mô hình hồi quy tuyến tính cho đặc trưng tốt nhất
reg_best = LinearRegression().fit(train_1[[best_feature_c]], train_1['Salary'])

# Dự đoán giá trị trên tập kiểm tra
predictions = reg_best.predict(test_1[[best_feature_c]])

# Tính Mean Absolute Error trên tập kiểm tra
mae_test = mean_absolute_error(test_1['Salary'], predictions)
print("Mean Absolute Error on test set:", mae_test)

# Hiển thị công thức hồi quy tuyến tính cho đặc trưng tốt nhất
intercept_best = reg_best.intercept_
linear_formula_best = f"{intercept_best:.2f} + {reg_best.coef_[0]:.2f} * {best_feature_c}"
print("Best linear formula:", linear_formula_best)