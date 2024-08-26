import numpy as np
import pandas as pd
import json
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# 在调试模式下设置日志级别为DEBUG
debug_mode = True  # 设置为True以启用调试模式
if debug_mode:
    logger.setLevel(logging.DEBUG)

class HardwareDeployment:
    def __init__(self, num_hardware_networks, hardware_bandwidth, input_data, sample_frequency, num_virtual_services):
        self.num_hardware_networks = num_hardware_networks
        self.hardware_bandwidth = hardware_bandwidth
        self.input_data = input_data
        self.sample_frequency = sample_frequency
        self.num_virtual_services = num_virtual_services
        self.queue_data_hardware = pd.DataFrame(columns=['REQ_NO', 'Size', 'Waiting_Time'])
        self.req_data_hardware = pd.DataFrame(columns=['REQ_NO', 'Arrival_Time', 'Start_Time', 'End_Time', 'Queue_Time',
                                                       'Delay_Time', 'Response_Time', 'Resource_Usage', 'Deployed', 'Hardware_Network'])
        self.time_data_hardware = pd.DataFrame(columns=['Time', 'Processing_Reqs', 'Num_of_Processing_Reqs',
                                                        'Processed_Reqs', 'Num_of_Processed_Reqs', 'Queuing_Reqs',
                                                        'Num_of_Queuing_Reqs', 'Refused_Reqs', 'Num_of_Refused_Reqs', 'Acceptance_Rate'])
        self.Num_of_Demands = 1  # 需求编号起始值
        self.Idle_flag_of_each_hardware_networks = [0] * self.num_hardware_networks  # 网络空闲标志位

    def process_current_demand_state(self, t):
        pre_overview_deployed = 0
        pre_overview_processing = 0
        pre_overview_queuing = 0
        pre_overview_outcast = 0
        for index, row in self.req_data_hardware.iterrows():
            if row['Deployed'] == 1:
                pre_overview_deployed += 1
                continue
            if row['End_Time'] <= t and not pd.isna(row['Start_Time']):
                self.req_data_hardware.loc[index, 'Queue_Time'] = row['Start_Time'] - row['Arrival_Time']
                self.req_data_hardware.loc[index, 'Response_Time'] = row['End_Time'] - row['Arrival_Time']
                self.req_data_hardware.loc[index, 'Deployed'] = 1
                self.Idle_flag_of_each_hardware_networks[self.req_data_hardware.loc[index, 'Hardware_Network']] = 0
                self.req_data_hardware.loc[index, 'Hardware_Network'] = 'None'
                pre_overview_deployed += 1
                continue
            if row['Delay_Time'] < t and pd.isna(row['Start_Time']):
                self.req_data_hardware.loc[index, 'Deployed'] = -1
                index_to_drop = self.queue_data_hardware[self.queue_data_hardware['REQ_NO'] == row['REQ_NO']].index
                self.queue_data_hardware = self.queue_data_hardware.drop(index_to_drop)
                pre_overview_outcast += 1
            pre_overview_queuing = len(self.queue_data_hardware)
            pre_overview_processing = len(self.req_data_hardware) - pre_overview_deployed - pre_overview_outcast - pre_overview_queuing

        logger.debug(
            f"当t={t}时，在分配需求至硬件网络之前，当前硬件网络有{pre_overview_deployed}个需求成功部署，有{pre_overview_processing}个需求正在执行，"
            f"有{pre_overview_queuing}个需求正在排队，有{pre_overview_outcast}个需求已被抛弃，"
            f"硬件网络{[zero_indices for zero_indices, x in enumerate(self.Idle_flag_of_each_hardware_networks) if x == 0]}"
            f"处于空闲状态，"
            f"硬件网络{[one_indices for one_indices, x in enumerate(self.Idle_flag_of_each_hardware_networks) if x == 1]}"
            f"处于占用状态!!!!!"
        )

    def handle_new_demand(self, t):
        if self.input_data['Time'].eq(t).any():
            self.queue_data_hardware.reset_index(drop=True, inplace=True)  # 重置索引
            Match_Row = self.input_data[self.input_data['Time'] == t].iloc[0]
            Match_Row['Size'] = json.loads(Match_Row['Size'])
            Match_Row['Delay'] = json.loads(Match_Row['Delay'])
            for i in range(Match_Row['Arrival Demand']):
                self.queue_data_hardware.loc[len(self.queue_data_hardware)] = [
                    int(self.Num_of_Demands),
                    Match_Row['Size'][i],
                    Match_Row['Delay'][i]
                ]
                self.req_data_hardware.loc[len(self.req_data_hardware), ['REQ_NO', 'Arrival_Time', 'Delay_Time']] = [self.Num_of_Demands, t, t + Match_Row['Delay'][i]]
                self.Num_of_Demands += 1

    def deploy_hardware(self, t):
        if self.queue_data_hardware.empty:
            print(f'在t={t}时刻,当前排队队列为空')
            return
        req_to_drop_hardware = []  # 用来标记被执行的需求，以此从queue中删除
        logger.debug(f"t={t},将Queue中的数据部署至{self.num_hardware_networks}个硬件网络中！")
        break_flag_for_the_next_t = False
        for index, row in self.queue_data_hardware.iterrows():
            for index_hard, value in enumerate(self.Idle_flag_of_each_hardware_networks):
                if all(value == 1 for value in self.Idle_flag_of_each_hardware_networks):
                    logger.debug(f"所有硬件网络均被占用！")
                    break_flag_for_the_next_t = True
                    break
                if value == 1:
                    logger.debug(f"在分配编号为{row['REQ_NO']}的需求时，部署至第{index_hard}个硬件网络，但该硬件网络正在执行别的需求故不行！")
                if value == 0:
                    logger.debug(f"在分配编号为{row['REQ_NO']}的需求时，成功部署至第{index_hard}个硬件网络")
                    self.Idle_flag_of_each_hardware_networks[index_hard] = 1
                    Duration_Time = row['Size'] / self.hardware_bandwidth * (self.num_virtual_services - 1)
                    self.req_data_hardware.loc[self.req_data_hardware['REQ_NO'] == int(row.iloc[0]), ['Start_Time', 'End_Time', 'Hardware_Network']] = [t, t + Duration_Time, index_hard]
                    req_to_drop_hardware.append(index)
                    break
            if break_flag_for_the_next_t:
                break

        self.queue_data_hardware = self.queue_data_hardware.drop(req_to_drop_hardware)  # 从队列中删除被部署的需求
        logger.debug(f"当t={t}时，队列中有{len(req_to_drop_hardware)}个需求被分配，有{self.queue_data_hardware.shape[0]}个需求还在队列中！")

    def build_time_based_statistics(self, t):
        processing_req_hardware = []
        processed_req_hardware = []
        refused_req_hardware = []
        queue_req_hardware = []
        num_processing_req_hardware = 0
        num_processed_req_hardware = 0
        num_refused_req_hardware = 0
        num_queue_req_hardware = 0

        for index, row in self.req_data_hardware.iterrows():
            if not pd.isna(row['Start_Time']) and pd.isna(row['Deployed']):
                num_processing_req_hardware += 1
                processing_req_hardware.append(row['REQ_NO'])
            elif not pd.isna(row['Start_Time']) and row['Deployed'] == 1:
                num_processed_req_hardware += 1
                processed_req_hardware.append(row['REQ_NO'])
            elif pd.isna(row['Start_Time']) and row['Deployed'] == -1:
                num_refused_req_hardware += 1
                refused_req_hardware.append(row['REQ_NO'])
            elif pd.isna(row['Start_Time']) and pd.isna(row['Deployed']):
                num_queue_req_hardware += 1
                queue_req_hardware.append(row['REQ_NO'])

        acceptance_rate_hardware = (num_processing_req_hardware + num_processed_req_hardware) / \
                                   (num_processing_req_hardware + num_processed_req_hardware + num_refused_req_hardware + num_queue_req_hardware)

        self.time_data_hardware.loc[self.time_data_hardware['Time'] == t, ['Processing_Reqs', 'Num_of_Processing_Reqs',
                                                                           'Processed_Reqs', 'Num_of_Processed_Reqs', 'Queuing_Reqs',
                                                                           'Num_of_Queuing_Reqs', 'Refused_Reqs', 'Num_of_Refused_Reqs', 'Acceptance_Rate']] = \
            [str(processing_req_hardware), num_processing_req_hardware, str(processed_req_hardware), num_processed_req_hardware,
             str(queue_req_hardware), num_queue_req_hardware, str(refused_req_hardware), num_refused_req_hardware, acceptance_rate_hardware]

    def finalize_unfinished_requests(self):
        for index, row in self.req_data_hardware.iterrows():
            if row['Deployed'] == 1 or row['Deployed'] == -1:
                continue
            if not pd.isna(row['Start_Time']):
                self.req_data_hardware.loc[index, 'Deployed'] = 1 
                self.req_data_hardware.loc[index, 'Queue_Time'] = row['Start_Time'] - row['Arrival_Time']
                self.req_data_hardware.loc[index, 'Response_Time'] = row['End_Time'] - row['Arrival_Time']
            else:
                self.req_data_hardware.loc[index, 'Deployed'] = -1e5
        rows_to_drop_hardware = self.req_data_hardware[self.req_data_hardware['Deployed'] == -1e5].index
        self.req_data_hardware = self.req_data_hardware.drop(rows_to_drop_hardware)
        counts_hardware = self.req_data_hardware['Deployed'].value_counts()
        # 获取等于 1 和 -1 的数量
        count_1_hardware = counts_hardware.get(1, 0)  # 如果没有 1 的项，会返回 0
        count_neg_1_hardware = counts_hardware.get(-1, 0)  # 如果没有 -1 的项，会返回 0
        logger.debug(f"有{count_1_hardware}个需求被部署，有{count_neg_1_hardware}个需求被舍弃！")

    def run(self):
        # 主循环，按时间步长处理需求
        for t in range(0, self.input_data['Time'].iloc[-1] + 1, self.sample_frequency):
            self.time_data_hardware.loc[len(self.time_data_hardware), ['Time']] = t  # 为time_data数据添加时刻

            self.process_current_demand_state(t)  # 判断当前需求状态
            self.handle_new_demand(t)  # 处理新到达的需求
            self.deploy_hardware(t)  # 开始硬件部署
            self.build_time_based_statistics(t)  # 构建时间为单位的统计数据

        self.finalize_unfinished_requests()  # 处理未完成的请求

        # 存储输出结果
        self.save_output_csv(self.req_data_hardware, "Hardware_data_ouput.csv")
        self.save_output_csv(self.time_data_hardware, "Hardware_time_output.csv")

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