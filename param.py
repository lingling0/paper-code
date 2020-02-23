import numpy as np

v2v_range = 200
T = 100  # 0.1s为1一个time slot
data_num = 10
# width = (right - left)/grid_width, 所以横格子最多27个，纵格子最多18个。格子数量为

grid_height, grid_width, grid_num = 100, 100, 486
bottom, top, left, right, width = 500, 2300, 1000, 3700, 54
data_length = np.random.poisson(lam=1, size=data_num)
data_length = [0.5 if i == 0 else i for i in data_length]
# print(data_length)
current_path = "C:/Users/26271/papercode/"
filename_data = "data/1/vehicle_data1.json"
request_file_name = "data/1/package_data1.json"
dag_file_name = "data/1/dag_data1.json"
