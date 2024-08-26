import pandas as pd
import numpy as np
import os
import json

# 获取当前脚本所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录
parent_directory = os.path.dirname(current_directory)
# 定义要检查的目录名称
data_folder_path = os.path.join(parent_directory, 'data')

# 创建一个随机数生成器
rng = np.random.default_rng(seed=7839267589) #设置种子后，每次生成一样的数据
# 生成序号列
sequence_numbers = np.arange(1, 201, 1)

# 生成时间间隔列，随机间隔在1到5之间
time_intervals = rng.integers(1, 4, size=200)#之前是(1,5)
# 生成时间列
time_column = np.cumsum(time_intervals) - time_intervals[0]
time_intervals[0] = 0
# 生成到达需求列，泊松分布，元素大小在2到6之间
arrival_demand = rng.poisson(7, size=200)
arrival_demand = np.clip(arrival_demand, 4, 10)#之前是(2,6)

#生成每个需求所产生的流量大小1mb~10mb
flow_avg_size_column = [np.round(rng.uniform(3, 10, size=demand), 2).tolist() for demand in arrival_demand]

#生成每个需求最长等待时间 
Delay_column = [np.round(rng.uniform(2, 4, size=demand), 2).tolist() for demand in arrival_demand]#之前是(1,4)

#生成每个需求资源占用情况
Resource_usage_column = [
    [[round(rng.uniform(1, 2), 2), round(rng.uniform(2, 4), 2), round(rng.uniform(2, 4), 2)] for _ in range(demand)]
    for demand in arrival_demand
]

# 创建数据框
df = pd.DataFrame({
    'No.': sequence_numbers,
    'Time': time_column,
    'Intervals': time_intervals,
    'Arrival Demand': arrival_demand,
    'Size': flow_avg_size_column,
    'Delay': Delay_column,
    'Resource': Resource_usage_column
})
df['Resource'] = df['Resource'].apply(lambda x: json.dumps(x, default=int))
# 保存到CSV文件
csv_path = os.path.join(data_folder_path, 'generated_data.csv')
df.to_csv(csv_path, index=False)

print(f"CSV file saved at: {csv_path}")