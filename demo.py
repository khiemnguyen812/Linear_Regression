import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Đọc dữ liệu từ file CSV
data = pd.read_csv('train.csv')

# Tách đặc trưng và biến mục tiêu
X = data.drop('Salary', axis=1)
y = data['Salary']

# Sử dụng mô hình Linear Regression làm mô hình cơ sở
base_model = LinearRegression()

# Khởi tạo RFE với mô hình cơ sở và số lượng đặc trưng muốn giữ lại
num_features_to_select = 3
rfe = RFE(estimator=base_model, n_features_to_select=num_features_to_select)

# Fit RFE vào dữ liệu
rfe.fit(X, y)

# Lựa chọn các đặc trưng tốt nhất
selected_features = X.columns[rfe.support_]
print("Selected features:", selected_features)
