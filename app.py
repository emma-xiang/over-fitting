from flask import Flask, render_template
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def generate_data():
    # 生成带有噪声的数据点
    np.random.seed(42)
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, X.shape)
    return X, y

def plot_model(X, y, degree, title):
    # 创建多项式特征
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # 训练模型
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # 生成平滑的预测线
    X_smooth = np.linspace(0, 1, 200).reshape(-1, 1)
    X_smooth_poly = poly_features.transform(X_smooth)
    y_smooth = model.predict(X_smooth_poly)
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='训练数据')
    plt.plot(X_smooth, y_smooth, color='red', label='模型预测')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    # 将图形转换为base64字符串
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    X, y = generate_data()
    
    # 生成正常拟合的模型（3次多项式）
    normal_fit = plot_model(X, y, 3, '正常拟合 (3次多项式)')
    
    # 生成过拟合的模型（15次多项式）
    overfit = plot_model(X, y, 15, '过度拟合 (15次多项式)')
    
    return render_template('index.html', 
                         normal_fit=normal_fit, 
                         overfit=overfit)

if __name__ == '__main__':
    app.run(debug=True)