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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# 在调试模式下设置日志级别为DEBUG
debug_mode = True  # 设置为True以启用调试模式
if debug_mode:
    logger.setLevel(logging.DEBUG)

class GreedyDeployment:
    def __init__(self, resource_usage, bandwidth_dict, input_data, sample_frequency, nodes_status):
        self.bandwidth_for_test = bandwidth_dict #各条链路的带宽
        self.input_data = input_data #输入数据
        self.sample_frequency = sample_frequency #采样频率
        self.num_virtual_services = len(resource_usage) #一个需求包含几个服务
        self.service_usage = resource_usage #每个服务所使用的资源情况
        self.nodes_status = nodes_status #各节点资源使用情况
        self.queue_data_greedy = pd.DataFrame(columns=['REQ_NO', 'Size', 'Waiting_Time'])
        self.req_data_greedy = pd.DataFrame(columns=['REQ_NO', 'Nodes', 'Arrival_Time',
                                                     'Start_Time', 'End_Time', 'Queue_Time',
                                                     'Delay_Time', 'Response_Time', 'Resource_Usage', 'Deployed'])
        self.time_data_greedy = pd.DataFrame(columns=['Time', 'Processing_Reqs', 'Num_of_Processing_Reqs',
                                                        'Processed_Reqs', 'Num_of_Processed_Reqs', 'Queuing_Reqs',
                                                        'Num_of_Queuing_Reqs', 'Refused_Reqs', 'Num_of_Refused_Reqs', 'Acceptance_Rate'])
        self.Num_of_Demands = 1  # 需求编号起始值

    def release_resource(self, req_nodes):
        req_nodes = json.loads(req_nodes)
        for j, k in enumerate(req_nodes):
            self.nodes_status[k] = self.nodes_status[k] + self.service_usage[j]

    def cal_lantency(self, gene, flow_size):
    # 计算传输时延
        total_latency = 0
        for m in range(len(gene) - 1):
            # 获取链路的起点和终点
            start_node = gene[m]
            end_node = gene[m + 1]
            # 查找链路的带宽
            if (start_node, end_node) in self.bandwidth_for_test:
                band = self.bandwidth_for_test[(start_node, end_node)]
            elif (end_node, start_node) in self.bandwidth_for_test:
                band = self.bandwidth_for_test[(end_node, start_node)]
            else:
                raise ValueError(f"在计算持续时间时，链路 ({start_node}, {end_node}) 没有定义带宽。")
            # 计算链路的传输时延
            latency = flow_size / band
            total_latency += latency
        return total_latency
    
    def process_current_demand_state(self, t):
        pre_overview_deployed = 0
        pre_overview_processing = 0
        pre_overview_queuing = 0
        pre_overview_outcast = 0
        for index, row in self.req_data_greedy.iterrows():
            if row['Deployed'] == 1:
                pre_overview_deployed += 1
                continue
            if row['End_Time'] <= t and not pd.isna(row['Start_Time']):
                self.req_data_greedy.loc[index, 'Queue_Time'] = row['Start_Time'] - row['Arrival_Time']
                self.req_data_greedy.loc[index, 'Response_Time'] = row['End_Time'] - row['Arrival_Time']
                self.req_data_greedy.loc[index, 'Deployed'] = 1
                pre_overview_deployed += 1
                 #释放占用资源
                self.release_resource(row['Nodes'])
                continue
            if row['Delay_Time'] < t and pd.isna(row['Start_Time']):
                self.req_data_greedy.loc[index, 'Deployed'] = -1
                index_to_drop = self.queue_data_greedy[self.queue_data_greedy['REQ_NO'] == row['REQ_NO']].index
                self.queue_data_greedy = self.queue_data_greedy.drop(index_to_drop)
                pre_overview_outcast += 1
            pre_overview_queuing = len(self.queue_data_greedy)
            pre_overview_processing = len(self.req_data_greedy) - pre_overview_deployed - pre_overview_outcast - pre_overview_queuing

        logger.debug(
            f"当t={t}时，在分配需求至CADA之前，当前有{pre_overview_deployed}个需求成功部署，有{pre_overview_processing}个需求正在执行，"
            f"有{pre_overview_queuing}个需求正在排队，有{pre_overview_outcast}个需求已被抛弃，"
            f"CADA中节点剩余资源情况为:{self.nodes_status}"
        )

    def handle_new_demand(self, t): #将新到来的需求放入排队队列中
        if self.input_data['Time'].eq(t).any():
            self.queue_data_greedy.reset_index(drop=True, inplace=True)  # 重置索引
            Match_Row = self.input_data[self.input_data['Time'] == t].iloc[0]
            Match_Row['Size'] = json.loads(Match_Row['Size'])
            Match_Row['Delay'] = json.loads(Match_Row['Delay'])
            for i in range(Match_Row['Arrival Demand']):
                self.queue_data_greedy.loc[len(self.queue_data_greedy)] = [
                    int(self.Num_of_Demands),
                    Match_Row['Size'][i],
                    Match_Row['Delay'][i]
                ]
                self.req_data_greedy.loc[len(self.req_data_greedy), ['REQ_NO', 'Arrival_Time', 'Delay_Time']] = [self.Num_of_Demands, t, t + Match_Row['Delay'][i]]
                self.Num_of_Demands += 1

    def greedy_deploy(self, t):
        if self.queue_data_greedy.empty:
            print(f'在t={t}时刻,当前排队队列为空')
            return
        req_to_drop_greedy = []  # 用来标记被执行的需求，以此从queue中删除
        logger.debug(f"t={t},将Queue中的需求{self.queue_data_greedy['REQ_NO'].tolist()}以贪心算法的形式部署至CADA中")
        for index, row in self.queue_data_greedy.iterrows():
            sorted_usage = sorted(self.service_usage.items(), key=lambda item: item[1], reverse=True)
            nodes_backup = {key: value for key, value in self.nodes_status.items() if value > 0}
            # 按节点剩余资源降序排序
            sorted_nodes = sorted(nodes_backup.items(), key=lambda item: item[1], reverse=True)
            current_allocation = {}
            can_allocate = True

            for index_usage, usage in sorted_usage:
                for node, available in sorted_nodes:
                    if available >= usage:
                        current_allocation[index_usage] = node #将最大资源占用的微服务分配给最大剩余资源的节点
                        nodes_backup[node] -= usage
                        del nodes_backup[node]  #一个节点只能分配一个微服务
                        sorted_nodes = sorted( nodes_backup.items(), key=lambda item: item[1], reverse=True)
                        break
                    else:
                        can_allocate = False
                        break
            
            if not can_allocate:
                continue

            sorted_keys = sorted(current_allocation.keys())
            req_nodes = {key: current_allocation[key] for key in sorted_keys}
            logger.debug(f"t={t}时，为编号{row['REQ_NO']}的需求分配的节点是{list(req_nodes.values())}")
            Duration_Time = self.cal_lantency(list(req_nodes.values()), row['Size'])
            self.req_data_greedy.loc[self.req_data_greedy['REQ_NO'] == int(row.iloc[0]), ['Start_Time', 'End_Time']] = [t, t + Duration_Time]
            self.req_data_greedy.loc[self.req_data_greedy['REQ_NO'] == int(row.iloc[0]), ['Nodes']] = str(list(req_nodes.values()))
            for j, k in enumerate(list(req_nodes.values())):
                self.nodes_status[k] = self.nodes_status[k] - self.service_usage[j]
            req_to_drop_greedy.append(index)

        self.queue_data_greedy = self.queue_data_greedy.drop(req_to_drop_greedy)  # 从队列中删除被部署的需求
        logger.debug(f"当t={t}时，队列中有{len(req_to_drop_greedy)}个需求被分配，有{self.queue_data_greedy.shape[0]}个需求还在队列中！")

    def build_time_based_statistics(self, t):
        processing_req_greedy = []
        processed_req_greedy  = []
        refused_req_greedy  = []
        queue_req_greedy  = []
        num_processing_req_greedy  = 0
        num_processed_req_greedy  = 0
        num_refused_req_greedy  = 0
        num_queue_req_greedy  = 0

        for index, row in self.req_data_greedy.iterrows():
            if not pd.isna(row['Start_Time']) and pd.isna(row['Deployed']):
                num_processing_req_greedy += 1
                processing_req_greedy.append(row['REQ_NO'])
            elif not pd.isna(row['Start_Time']) and row['Deployed'] == 1:
                num_processed_req_greedy += 1
                processed_req_greedy.append(row['REQ_NO'])
            elif pd.isna(row['Start_Time']) and row['Deployed'] == -1:
                num_refused_req_greedy += 1
                refused_req_greedy.append(row['REQ_NO'])
            elif pd.isna(row['Start_Time']) and pd.isna(row['Deployed']):
                num_queue_req_greedy += 1
                queue_req_greedy.append(row['REQ_NO'])

        acceptance_rate_greedy = (num_processing_req_greedy + num_processed_req_greedy) / \
                                   (num_processing_req_greedy + num_processed_req_greedy + num_refused_req_greedy + num_queue_req_greedy)

        self.time_data_greedy.loc[self.time_data_greedy['Time'] == t, ['Processing_Reqs', 'Num_of_Processing_Reqs',
                                                                           'Processed_Reqs', 'Num_of_Processed_Reqs', 'Queuing_Reqs',
                                                                           'Num_of_Queuing_Reqs', 'Refused_Reqs', 'Num_of_Refused_Reqs', 'Acceptance_Rate']] = \
            [str(processing_req_greedy), num_processing_req_greedy, str(processed_req_greedy), num_processed_req_greedy,
             str(queue_req_greedy), num_queue_req_greedy, str(refused_req_greedy), num_refused_req_greedy, acceptance_rate_greedy]

    def finalize_unfinished_requests(self):
        for index, row in self.req_data_greedy.iterrows():
            if row['Deployed'] == 1 or row['Deployed'] == -1:
                continue
            if not pd.isna(row['Start_Time']):
                self.req_data_greedy.loc[index, 'Deployed'] = 1 
                self.req_data_greedy.loc[index, 'Queue_Time'] = row['Start_Time'] - row['Arrival_Time']
                self.req_data_greedy.loc[index, 'Response_Time'] = row['End_Time'] - row['Arrival_Time']
            else:
                self.req_data_greedy.loc[index, 'Deployed'] = -1e5
        rows_to_drop_greedy = self.req_data_greedy[self.req_data_greedy['Deployed'] == -1e5].index
        self.req_data_greedy = self.req_data_greedy.drop(rows_to_drop_greedy)
        counts_greedy = self.req_data_greedy['Deployed'].value_counts()
        # 获取等于 1 和 -1 的数量
        count_1_greedy = counts_greedy.get(1, 0)  # 如果没有 1 的项，会返回 0
        count_neg_1_greedy = counts_greedy.get(-1, 0)  # 如果没有 -1 的项，会返回 0
        logger.debug(f"有{count_1_greedy}个需求被部署，有{count_neg_1_greedy}个需求被舍弃！")

    def run(self):
        # 主循环，按时间步长处理需求
        for t in range(0, self.input_data['Time'].iloc[-1] + 1, self.sample_frequency):
            self.time_data_greedy.loc[len(self.time_data_greedy), ['Time']] = t  # 为time_data数据添加时刻

            self.process_current_demand_state(t)  # 判断当前需求状态
            self.handle_new_demand(t)  # 处理新到达的需求
            self.greedy_deploy(t)  # 开始硬件部署
            self.build_time_based_statistics(t)  # 构建时间为单位的统计数据

        self.finalize_unfinished_requests()  # 处理未完成的请求

        # 存储输出结果
        self.save_output_csv(self.req_data_greedy, "greedy_without_weights_data_ouput.csv")
        self.save_output_csv(self.time_data_greedy, "greedy_without_weights_time_output.csv")

    def save_output_csv(self, output_csv, output_name):
        # 获取当前脚本所在的绝对路径
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # 获取上一级目录
        parent_directory = os.path.dirname(current_directory)

        # 定义要检查的目录名称
        data_folder_path = os.path.join(parent_directory, 'data')

        # 检查data文件夹是否存在
        if not os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
            raise FileNotFoundError("data文件夹不存在,请去上一级目录新建一个data文件夹,并将输入数据放入其中！")

        # 存储csv文件
        csv_path = os.path.join(data_folder_path, output_name)
        output_csv.to_csv(csv_path, index=True)