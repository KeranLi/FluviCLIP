import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_predictions(actual, predicted, title, filename):
    """
    绘制实际值与预测值的散点图。

    Args:
        actual (list or np.array): 实际值。
        predicted (list or np.array): 预测值。
        title (str): 图表标题。
        filename (str): 保存图片的文件名。
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(actual)), actual, label="Actual", marker='o', color='blue', alpha=0.7)
    plt.scatter(range(len(predicted)), predicted, label="Predicted", marker='x', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

#def plot_loss_curves(train_losses, val_losses, test_losses, output_dir, filename="loss_curve.png"):
    """
    绘制训练、验证和测试损失曲线，并保存为图片。

    Args:
        train_losses (list): 每个 epoch 的训练损失列表。
        val_losses (list): 每个 epoch 的验证损失列表。
        test_losses (list): 每个 epoch 的测试损失列表。
        output_dir (str): 保存图片的输出目录。
        filename (str): 图片文件名，默认为 "loss_curve.png"。
    """
    #plt.figure(figsize=(12, 6))
    #plt.plot(train_losses, label="Training Loss")
    #plt.plot(val_losses, label="Validation Loss")
    #plt.plot(test_losses, label="Test Loss")
    #plt.xlabel("Epoch")
    #plt.ylabel("Loss")
    #plt.title("Training, Validation, and Test Loss Over Epochs")
    #plt.legend()
    #plt.grid(True)
    #loss_curve_file = os.path.join(output_dir, filename)
    #plt.savefig(loss_curve_file)
    #plt.close()

def plot_loss_curves(train_losses, val_losses, test_losses, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, Validation, and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()    
    
def plot_actual_vs_pred(actual, predicted, title, filename, xlim=(0, 2000), ylim=(0, 2000)):
    """
    绘制预测值与实际值的散点图，过滤掉小于0的值，并添加 y=x 的虚线和拟合直线。
    同时将拟合公式直接显示在图像中。

    Args:
        actual (list or np.array): 实际值。
        predicted (list or np.array): 预测值。
        title (str): 图表标题。
        filename (str): 保存图片的文件名。
        xlim (tuple): x 轴的范围。
        ylim (tuple): y 轴的范围。
    """
    # 转换为 NumPy 数组
    actual = np.array(actual)
    predicted = np.array(predicted)

    # 过滤小于0的值
    mask = (actual > 0) & (predicted > 0)
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]

    # 拟合直线
    predicted_filtered = predicted_filtered.reshape(-1, 1)  # 转换为二维数组以适配 LinearRegression
    reg = LinearRegression().fit(predicted_filtered, actual_filtered)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    line_x = np.linspace(xlim[0], xlim[1], 500)  # 生成拟合直线的 x 轴数据
    line_y = slope * line_x + intercept

    # 绘图
    plt.figure(figsize=(8, 8))
    plt.scatter(predicted_filtered, actual_filtered, color='blue', alpha=0.7, label="Predicted vs Actual")
    plt.plot(line_x, line_y, color='green', linestyle='-', linewidth=2, label="Fit Line")
    plt.plot(xlim, ylim, linestyle='--', color='red', label="y = x")
    plt.title(title)
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # 保证横纵坐标比例一致

    # 在图像中添加拟合公式
    formula_text = f"y = {slope:.2f}x + {intercept:.2f}"
    plt.text(
        0.05 * xlim[1], 0.9 * ylim[1],  # 设置公式的位置
        formula_text, 
        fontsize=12, 
        color='green', 
        bbox=dict(facecolor='white', alpha=0.5)
    )

    # 保存图片
    plt.savefig(filename)
    plt.close()
