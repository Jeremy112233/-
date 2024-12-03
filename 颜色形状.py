# author: Jeremy
# crate-data:2024.5.24

import cv2
import numpy as np
import glob

# 定义颜色范围
color_ranges = {
    'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
    'green': (np.array([60, 100, 100]), np.array([90, 255, 255])),
    'blue': (np.array([90, 100, 100]), np.array([120, 255, 255]))
}

# # 定义形状检测函数
# def detect_shape(contour):
#     peri = cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
#     if len(approx) == 3:
#         return "Triangle"
#     elif len(approx) == 4:
#         return "Square"
#     else:
#         return "Circle"
def detect_shape(contour):
    # 计算轮廓的周长
    peri = cv2.arcLength(contour, True)
    # 轮廓近似
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    # 获取轮廓的边界盒
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h

    # 根据顶点数量和宽高比判断形状
    if len(approx) == 3:
        # 三角形
        return "Triangle"
    elif len(approx) == 4:
        # 正方形通常具有接近1的宽高比
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            # 可能是矩形或其他四边形
            return "Quadrilateral"
    elif len(approx) > 6:
        # 如果顶点数量大于4且轮廓是凸的，可能是一个圆形
        # 这里只是作为一个可能的启发式方法，实际情况可能需要更精确的方法来判断
        return "Circle"
    else:
        return "Square"
    # 高斯噪声
def gasuss_noise(image, mean=0, var=0.01):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var**0.5, image.shape)
    img_noise = image + noise
    if img_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    img_noise = np.clip(img_noise, low_clip, 1.0)
    img_noise = np.uint8(img_noise * 255)
    return img_noise


# 定义计数器变量
color_counts = {color: 0 for color in color_ranges.keys()}
shape_counts = {shape: 0 for shape in ['Triangle', 'Square', 'Circle']}
color_shape_counts = {}
sum1 = 0

# 定义要处理的图像路径
image_paths = glob.glob("E:\\pythonProject\\cv\\photo1\\*.PNG")

# 遍历图像路径
for path in image_paths:

    # 读取图像
    print(path)
    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 初始化结果图像
    result = image.copy()

    # 加噪
    img_gasuss = gasuss_noise(result, mean=0, var=0.01)  # 高斯噪声
    cv2.imshow("add_noise_wjm", img_gasuss)
    # 去噪
    img_gasuss_out = cv2.GaussianBlur(img_gasuss, (15, 15), 0, 0)
    cv2.imshow("Denoising__wjm", img_gasuss_out)

    # 开运算
    result1 = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(result1, cv2.MORPH_OPEN, kernel)
    cv2.imshow("open_operation_wjm", opening)
    # 闭运算
    closing = cv2.morphologyEx(result1, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("close_operation_wjm", closing)

    # 遍历颜色和形状
    colors = ['red', 'green', 'blue']
    shapes = ['Triangle', 'Square', 'Circle']
    np.random.shuffle(colors)
    np.random.shuffle(shapes)

    # 初始化组合计数器
    color_shape_counts[path] = {}
    for i in range(len(colors)):
        color = colors[i]
        shape = shapes[i]

        # 根据颜色范围创建掩码
        lower_range, upper_range = color_ranges[color]
        mask = cv2.inRange(hsv, lower_range, upper_range)
        # cv2.imshow("mask", mask)

        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历轮廓并绘制结果
        for contour in contours:
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
            shape_name = detect_shape(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, f"{color} {shape_name}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                print("检测出：", color, shape_name)

            # 更新颜色和形状组合的计数
            color_shape_key = f"{color} {shape_name}"
            if color_shape_key not in color_shape_counts[path]:
                color_shape_counts[path][color_shape_key] = 0
            color_shape_counts[path][color_shape_key] += 1

    # 计算价格
    prices = {
        'red Circle': 3,
        'blue Square': 5,
        'green Triangle': 7,
        'blue Circle': 4,
        'red Square': 6
    }
    total_price = 0
    print(color_shape_counts[path])
    counts = color_shape_counts[path]
    for color_shape, count in counts.items():
        if color_shape in prices:
            total_price += prices[color_shape] * count

    # 在图片上显示价格
    cv2.putText(result, f"Total Price: {total_price}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    print("彩色积木价格：", total_price)

    # 显示结果
    result = cv2.resize(result, None, fx=3/4, fy=3/4)
    cv2.imshow("Result", result)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# # 显示每种颜色和形状组合的数量
# for path, counts in color_shape_counts.items():
#     sum1 = 0
#     print(f"在 {path} 中的颜色和形状数量:")
#     for color_shape, count in counts.items():
#         if color_shape == "red Circle":
#             sum1 = sum1 + 3*count
#         if color_shape == "blue Square":
#             sum1 = sum1 + 5*count
#         if color_shape == "green Triangle":
#             sum1 = sum1 + 7*count
#         if color_shape == "blue Circle":
#             sum1 = sum1 + 4*count
#         if color_shape == "red Square":
#             sum1 = sum1 + 6*count
#         print(f"  {color_shape} 的数量为: {count}")
#     print("彩色积木价格：", sum1)

