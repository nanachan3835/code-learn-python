import matplotlib.pyplot as plt
import numpy as np
data1 = np.loadtxt('./Q-deep-q/data-giul39/AR_deep_q_learning.txt')*100
data2 = np.loadtxt('./Q-deep-q/data-giul39/AR_q_learning.txt')*100

mean_value1 = np.mean(data1)
mean_value2 = np.mean(data2)
np.std()

# Vẽ đồ thị cho dữ liệu mảng thứ nhất với màu xanh (blue)
plt.plot(data1, color='blue', linewidth=3, label='Deep Q-learning')
plt.axhline(mean_value1, color='blue', linewidth=3)

# Vẽ đồ thị cho dữ liệu mảng thứ hai với màu đỏ (red)
plt.plot(data2, color='red', linewidth=3, label='Q-learning')
plt.axhline(mean_value2, color='blue', linewidth=3)

# Đặt tiêu đề cho đồ thị
plt.title('Biểu đồ dữ liệu acceptation rate cho mạng giul39')

# Đặt nhãn cho trục x và y
plt.xlabel('SFC request')
plt.ylabel('Acceptation Rate (%)')

# Hiển thị chú thích với tên dữ liệu tương ứng
plt.legend()
# Hiển thị đồ thị
plt.show()
