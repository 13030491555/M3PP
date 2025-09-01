"""
该脚本用于分析显微图像中的颗粒特征。
它能自动识别图像中的暗色颗粒，计算它们的尺寸、位置，
并输出关键的统计指标，同时生成带有标记的可视化结果图。
"""

import cv2
import numpy as np
from scipy.spatial import distance
import sys

# --- 1. 配置常量 ---
# 修改这些值即可调整脚本行为，无需改动主逻辑。

# 文件路径配置
INPUT_IMAGE_PATH = '1.tif'
OUTPUT_IMAGE_ALL_MARKINGS = 'output_image_with_all_markings.png'
OUTPUT_IMAGE_CIRCLES_ONLY = 'output_image_with_circles_only.png'

# 图像处理参数
BINARY_THRESHOLD = 127  # 二值化阈值 (0-255)
MIN_PARTICLE_AREA = 10  # 识别为有效颗粒的最小面积（像素）

# 可视化参数
CENTROID_COLOR_BGR = (0, 0, 255)  # 质心标记颜色 (B, G, R) -> 红色
ENCLOSING_CIRCLE_COLOR_BGR = (0, 255, 0)  # 外接圆颜色 (B, G, R) -> 绿色
MARKER_RADIUS = 5  # 质心标记的半径
CIRCLE_THICKNESS = 2  # 外接圆的线宽


def load_and_preprocess(image_path: str, threshold_value: int) -> np.ndarray:
    """
    加载图像，进行灰度化和反向二值化处理。

    Args:
        image_path (str): 输入图像的文件路径。
        threshold_value (int): 用于二值化的灰度阈值。

    Returns:
        np.ndarray: 处理后的二值化图像。如果图像加载失败，则程序退出。
    """
    print(f"INFO: 正在从 '{image_path}' 加载图像...")
    # 以灰度模式加载图像
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否成功加载
    if gray_image is None:
        print(f"错误：无法在路径 '{image_path}' 找到或打开图像。请检查路径是否正确。")
        sys.exit(1)  # 错误退出

    # 对图像进行反向二值化处理
    # 灰度值低于阈值的像素（暗色）变为白色（255），其余变为黑色（0）
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_image


def extract_particle_features(binary_image: np.ndarray, min_area: float) -> tuple:
    """
    从二值图像中查找轮廓，并提取有效颗粒的面积、质心和半径。

    Args:
        binary_image (np.ndarray): 输入的二值图像。
        min_area (float): 判定为有效颗粒的最小面积。

    Returns:
        tuple: 包含三个列表的元组 (areas, centroids, radii)。
    """
    print("INFO: 正在查找轮廓并提取特征...")
    # 只检测最外层的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas, centroids, radii = [], [], []
    for contour in contours:
        area = cv2.contourArea(contour)
        # 过滤掉面积过小的噪点
        if area > min_area:
            M = cv2.moments(contour)
            # 确保面积不为零以避免除法错误
            if M["m00"] != 0:
                # 计算质心
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 计算最小外接圆
                _, radius = cv2.minEnclosingCircle(contour)

                # 存储提取到的特征
                areas.append(area)
                centroids.append((cX, cY))
                radii.append(radius)

    print(f"INFO: 检测到 {len(centroids)} 个有效颗粒。")
    return areas, centroids, radii


def analyze_and_print_stats(areas: list, radii: list, centroids: list):
    """
    计算并打印颗粒特征的详细统计数据。

    Args:
        areas (list): 颗粒面积列表。
        radii (list): 颗粒半径列表。
        centroids (list): 颗粒质心坐标列表。
    """
    print("\n--- 颗粒特征分析报告 ---")

    # 首先打印基础信息
    total_area = sum(areas)
    print(f"总面积 (Total Area): {total_area:.2f} 像素")
    print("-" * 30)

    # 必须保证有两个以上的颗粒才能进行距离和标准差的计算
    if len(radii) < 2:
        print("警告：有效颗粒不足两个，无法进行完整的统计分析。")
        if len(radii) == 1:
            print(f"平均半径 (Mean Radius): {radii[0]:.4f}")
        return

    # 1. 计算半径的平均值和标准差
    mean_radius = np.mean(radii)
    std_radius = np.std(radii)

    # 2. 计算所有质心点两两之间的欧氏距离
    dist_matrix = distance.cdist(centroids, centroids, 'euclidean')
    # 将对角线的0（自己到自己的距离）替换为无穷大，以便找到真正的最近邻
    np.fill_diagonal(dist_matrix, np.inf)
    # 对每个点找出其最近邻的距离
    nearest_distances = np.min(dist_matrix, axis=1)

    # 3. 计算最近邻距离的平均值和标准差
    mean_distance = np.mean(nearest_distances)
    std_distance = np.std(nearest_distances)

    # 4. 打印最终的关键指标结果
    print(f"平均半径 (meanRadius):   {mean_radius:.9f}")
    print(f"半径标准差 (stdRadius):    {std_radius:.9f}")
    print(f"平均最近邻距离 (meanDistance): {mean_distance:.9f}")
    print(f"最近邻距离标准差 (stdDistance):  {std_distance:.9f}")


def generate_visualizations(binary_image: np.ndarray, centroids: list, radii: list):
    """
    在图像上绘制标记并保存结果。

    Args:
        binary_image (np.ndarray): 用于绘制的基底二值图。
        centroids (list): 颗粒质心列表。
        radii (list): 颗粒半径列表。
    """
    print("\nINFO: 正在生成可视化结果图...")
    # 创建两个用于可视化的彩色输出图像副本
    all_markings_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    circles_only_image = all_markings_image.copy()

    # 遍历所有有效颗粒并绘制标记
    for i in range(len(centroids)):
        center = centroids[i]
        radius = radii[i]

        # 在第一张图上绘制质心和外接圆
        cv2.circle(all_markings_image, center, MARKER_RADIUS, CENTROID_COLOR_BGR, -1)  # -1表示实心
        cv2.circle(all_markings_image, center, int(radius), ENCLOSING_CIRCLE_COLOR_BGR, CIRCLE_THICKNESS)

        # 在第二张图上只绘制外接圆
        cv2.circle(circles_only_image, center, int(radius), ENCLOSING_CIRCLE_COLOR_BGR, CIRCLE_THICKNESS)

    # 保存两张结果图像
    cv2.imwrite(OUTPUT_IMAGE_ALL_MARKINGS, all_markings_image)
    cv2.imwrite(OUTPUT_IMAGE_CIRCLES_ONLY, circles_only_image)
    print(f"INFO: 结果图已保存至 '{OUTPUT_IMAGE_ALL_MARKINGS}' 和 '{OUTPUT_IMAGE_CIRCLES_ONLY}'。")


def main():
    """主函数，编排整个处理流程。"""
    # 步骤1：加载和预处理图像
    binary_image = load_and_preprocess(INPUT_IMAGE_PATH, BINARY_THRESHOLD)

    # 步骤2：提取颗粒特征
    areas, centroids, radii = extract_particle_features(binary_image, MIN_PARTICLE_AREA)

    # 步骤3：计算并打印统计分析结果
    analyze_and_print_stats(areas, radii, centroids)

    # 步骤4：生成并保存可视化图像
    if centroids:  # 仅在检测到颗粒时才生成图像
        generate_visualizations(binary_image, centroids, radii)

    print("\n--- 任务完成 ---")


if __name__ == "__main__":
    main()

