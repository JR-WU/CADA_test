from Hardware_deployment import HardwareDeployment
from randomdeployment import RandomDeploymentWithoutWeights
from GreedyDeployment import GreedyDeployment
import numpy as np
import pandas as pd
from scipy.stats import poisson, expon
import random
import csv
import os
import logging
import math
import copy
import json


def read_input_csv(input_name, is_skip_first_column):
    # 获取当前脚本所在的绝对路径
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # 获取上一级目录
    parent_directory = os.path.dirname(current_directory)
    
    # 定义要检查的目录名称
    data_folder_path = os.path.join(parent_directory, 'data')
    
    # 检查data文件夹是否存在
    if not os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        raise FileNotFoundError("data文件夹不存在,请去上一级目录新建一个data文件夹,并将输入数据放入其中！")

    #读取csv文件
    csv_path = os.path.join(data_folder_path, input_name)
    input_data = pd.read_csv(csv_path)
    if is_skip_first_column and input_data.columns[0] == 'No.':
        input_data = input_data.drop(input_data.columns[0], axis = 1)
    return input_data

def create_cada_nodes(NumOfHardNet, NumOfNodes, resource):
    #计算字典的大小
    num_keys = NumOfHardNet * NumOfNodes
    #创建字典，每个key对应的值都是resource
    cada_nodes = {i: resource for i in range(num_keys)}
    return cada_nodes

def check_networking_setting(settings, num_virtual_services, Number_of_Hardware_Networks):
    # 获取当前脚本所在的绝对路径
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # 获取上一级目录
    parent_directory = os.path.dirname(current_directory)
    
    # 定义要检查的目录名称
    data_folder_path = os.path.join(parent_directory, 'data')

    settings_path = os.path.join(data_folder_path, settings)
    
    # 检查data文件夹是否存在
    if not os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        raise FileNotFoundError("data文件夹不存在,请去上一级目录新建一个data文件夹,并将输入数据放入其中！")
    
    #检查data文件中是否存在settings.json, 不能存在则创建一个
    bandwidth_data = {}
    if not os.path.exists(settings_path):
        for i in range(num_virtual_services * Number_of_Hardware_Networks):
            for j in range(i + 1, num_virtual_services * Number_of_Hardware_Networks):
                # 随机生成5到10mb的带宽
                bandwidth = random.randint(5, 10)
                # 使用元组作为键，存储节点对的带宽值
                bandwidth_data[(i, j)] = bandwidth
        with open(settings_path, 'w') as f:
            for key, value in bandwidth_data.items():
                f.write(f"{key[0]} {key[1]} {value}\n")
        return bandwidth_data
    else:
        with open(settings_path, 'r') as f:
            for line in f:
                # 解析每一行，恢复为 (i, j): value 格式
                i, j, value = map(int, line.split())
                bandwidth_data[(i, j)] = value
        return bandwidth_data

#参数设置
Number_of_Hardware_Networks = 3 #硬件网络个数
hardware_bandwidth =5 
cada_node_resource = 5
input_data = read_input_csv('generated_data.csv', True) #读取输入数据
sample_frequency = 1
nodes_of_each_network = 3
each_service_usage = [1,2,3] #每个服务所用的资源，可以随机也可以固定。

#初始化网络、节点、服务
service_usage = {i: None for i in range(len(each_service_usage))}
for i in service_usage.keys():
    service_usage[i] = each_service_usage[i]
NodesCada = create_cada_nodes(Number_of_Hardware_Networks, nodes_of_each_network, cada_node_resource)
bandwidth_dict = check_networking_setting('settings.txt', nodes_of_each_network, Number_of_Hardware_Networks)

# ! 硬件部署

# 实例化HardwareDeployment类
hardware_deployment = HardwareDeployment(
    num_hardware_networks=Number_of_Hardware_Networks,
    hardware_bandwidth=hardware_bandwidth,
    input_data=input_data,
    sample_frequency=sample_frequency,
    num_virtual_services = nodes_of_each_network
)

# ! 运行硬件部署
# hardware_deployment.run()

# # * 随机部署
# #实例化RandomDeploymentWithoutWeights类
# random_deployment_without_weights = RandomDeploymentWithoutWeights(
#     resource_usage = service_usage,
#     bandwidth_dict = bandwidth_dict,
#     input_data = input_data,
#     sample_frequency = sample_frequency,
#     nodes_status = NodesCada
# )

# # * 运行不加权的随机部署
# random_deployment_without_weights.run()

# ? 贪心算法部署
greedy_deployment = GreedyDeployment(
    resource_usage = service_usage,
    bandwidth_dict = bandwidth_dict,
    input_data = input_data,
    sample_frequency = sample_frequency,
    nodes_status = NodesCada
)

# ? 运行贪心算法部署
greedy_deployment.run()