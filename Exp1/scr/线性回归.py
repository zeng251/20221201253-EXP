import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

# 设置中文字体以避免警告并正确显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示


# 1. 数据预处理函数：划分训练集和测试集
def getTrainSetAndTestSet(DataPath):
    # 检查文件是否存在且可读
    if not os.path.exists(DataPath):
        raise FileNotFoundError(f"文件不存在：{DataPath}")
    if not os.access(DataPath, os.R_OK):
        raise PermissionError(f"没有读取权限：{DataPath}")

    data = pd.read_csv(DataPath)
    X = data[['AT', 'V', 'AP', 'RH']]  # 特征数据
    y = data[['PE']]  # 标签数据

    # 按3:1比例划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=0.25
    )

    # 查看训练集和测试集的维度
    print("训练集特征维度：", X_train.shape)
    print("训练集标签维度：", y_train.shape)
    print("测试集特征维度：", X_test.shape)
    print("测试集标签维度：", y_test.shape)

    return X_train, X_test, y_train, y_test


# 2. 训练线性回归模型函数
def TrainLinearRegression(X_train, y_train):
    linreg = LinearRegression()  # 初始化模型
    linreg.fit(X_train, y_train)  # 训练模型

    # 输出截距和系数
    print("\n线性回归截距(θ0)：", linreg.intercept_[0])
    print("线性回归系数(θ1-θ4)：", linreg.coef_[0])
    print("回归方程：PE = {:.4f} + {:.4f}*AT + {:.4f}*V + {:.4f}*AP + {:.4f}*RH".format(
        linreg.intercept_[0], *linreg.coef_[0]
    ))

    return linreg


# 3. 评估模型性能函数
def EvaluationModel(linreg, X_test, y_test):
    y_pred = linreg.predict(X_test)  # 预测测试集

    # 计算均方误差（MSE）和均方根误差（RMSE）
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)

    print("\n均方误差（MSE）：", mse)
    print("均方根误差（RMSE）：", rmse)

    return y_pred


# 4. 可视化函数
def Visualization(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, color='blue')  # 绘制散点图
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)  # 绘制对比虚线

    # 设置中文标题和轴标签
    ax.set_xlabel("实际值（Measured）", fontsize=12)
    ax.set_ylabel("预测值（Predicted）", fontsize=12)
    ax.set_title("预测值与实际值对比", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
    plt.tight_layout()  # 优化布局
    plt.show()


# 主函数：执行完整流程
if __name__ == "__main__":
    # 使用原始字符串 r"..." 避免转义问题
    DataPath = r"C:\Users\86130\Desktop\线性回归.csv"  # 修改为你的文件路径

    try:
        # 验证文件权限
        print(f"正在检查文件：{DataPath}")
        if not os.path.exists(DataPath):
            print(f"错误：文件不存在 - {DataPath}")
            print(f"目录内容：{os.listdir(os.path.dirname(DataPath))}")
            exit(1)

        if not os.access(DataPath, os.R_OK):
            print(f"错误：没有读取权限 - {DataPath}")
            print("请尝试以下操作：")
            print("1. 以管理员身份运行 PyCharm")
            print("2. 将文件移动到用户文档目录")
            exit(1)

        # 执行正常流程
        print("开始数据预处理...")
        X_train, X_test, y_train, y_test = getTrainSetAndTestSet(DataPath)

        print("\n开始训练线性回归模型...")
        model = TrainLinearRegression(X_train, y_train)

        print("\n开始评估模型性能...")
        y_pred = EvaluationModel(model, X_test, y_test)

        print("\n生成可视化图表...")
        Visualization(y_test, y_pred)

        print("\n程序执行完毕！")

    except FileNotFoundError as e:
        print(f"文件错误：{e}")
    except PermissionError as e:
        print(f"权限错误：{e}")
        print("解决方案：")
        print("1. 关闭所有打开该文件的程序")
        print("2. 右键文件 → 属性 → 安全 → 确保用户有读取权限")
        print("3. 以管理员身份运行 PyCharm")
    except Exception as e:
        print(f"发生未知错误：{e}")