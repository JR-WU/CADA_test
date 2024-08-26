import numpy as np
import pandas as pd
from scipy.stats import poisson, expon
import random
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

def read_input_csv(input_name, is_skip_first_column):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    data_folder_path = os.path.join(parent_directory, 'data')

    if not os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        raise FileNotFoundError("data文件夹不存在,请去上一级目录新建一个data文件夹,并将输入数据放入其中！")

    csv_path = os.path.join(data_folder_path, input_name)
    input_data = pd.read_csv(csv_path)
    if is_skip_first_column and input_data.columns[0] == 'No.':
        input_data = input_data.drop(input_data.columns[0], axis=1)
    return input_data

def save_output_csv(output_csv, output_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    data_folder_path = os.path.join(parent_directory, 'data')

    if not os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        raise FileNotFoundError("data文件夹不存在,请去上一级目录新建一个data文件夹,并将输入数据放入其中！")

    csv_path = os.path.join(data_folder_path, output_name)
    output_csv.to_csv(csv_path, index=True)

def create_cada_nodes(NumOfHardNet, NumOfStep, resource):
    num_keys = NumOfHardNet * NumOfStep
    return {i: resource for i in range(num_keys)}

def create_resource_usage(num_ser, res_use):
    if len(res_use) != num_ser:
        raise ValueError("输入的资源使用数组长度与定义的虚拟化服务个数不匹配！")
    return res_use

def release_resource(req_nodes, resource_usage, nodes):
    req_nodes = json.loads(req_nodes)
    for j, k in enumerate(req_nodes):
        nodes[k] = nodes[k] + resource_usage[j]

def check_networking_setting(settings, num_virtual_services, Number_of_Hardware_Networks):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    data_folder_path = os.path.join(parent_directory, 'data')
    settings_path = os.path.join(data_folder_path, settings)

    if not os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        raise FileNotFoundError("data文件夹不存在,请去上一级目录新建一个data文件夹,并将输入数据放入其中！")
    
    bandwidth_data = {}
    if not os.path.exists(settings_path):
        for i in range(num_virtual_services * Number_of_Hardware_Networks):
            for j in range(i + 1, num_virtual_services * Number_of_Hardware_Networks):
                bandwidth = random.randint(5, 10)
                bandwidth_data[(i, j)] = bandwidth
        with open(settings_path, 'w') as f:
            for key, value in bandwidth_data.items():
                f.write(f"{key[0]} {key[1]} {value}\n")
        return bandwidth_data
    else:
        with open(settings_path, 'r') as f:
            for line in f:
                i, j, value = map(int, line.split())
                bandwidth_data[(i, j)] = value
        return bandwidth_data

def calc_latency(gene, bandwidth, flow_size):
    total_latency = 0
    for m in range(len(gene) - 1):
        start_node, end_node = gene[m], gene[m + 1]
        band = bandwidth.get((start_node, end_node)) or bandwidth.get((end_node, start_node))
        if not band:
            raise ValueError(f"链路 ({start_node}, {end_node}) 没有定义带宽。")
        total_latency += flow_size / band
    return total_latency

def check_current_nodes_suit_for_current_req(nodes, usage):
    temp_nodes = list(nodes.values())
    usage_backup = sorted(usage, reverse=True)
    can_satisfy = True

    for u in usage_backup:
        possible = False
        for i in range(len(temp_nodes)):
            if temp_nodes[i] >= u:
                temp_nodes[i] -= u
                possible = True
                break
        if not possible:
            can_satisfy = False
            break
    return can_satisfy

# 环境参数初始化
num_virtual_services = 3
Number_of_Hardware_Networks = 3
resource_of_each_node = 5
sample_frequency = 1
REQ_NO = 'Request_Number'
score_min = 0
score_max = 1.5

# 遗传算法参数
pop_size = 30
max_gen = 1
mutation_rate = 0.01
gama = 0.5

# 读取、创建实验需要的数据
input_data = read_input_csv('generated_data.csv', True)
virtual_nodes = create_cada_nodes(Number_of_Hardware_Networks, num_virtual_services, resource_of_each_node)
bandwidth_pair_nodes = check_networking_setting('settings.txt', num_virtual_services, Number_of_Hardware_Networks)
symmetric_bandwidth_data = bandwidth_pair_nodes.copy()
for (i, j), value in bandwidth_pair_nodes.items():
    symmetric_bandwidth_data[j, i] = value

class GeneticAlgorithmCada:
    def __init__(self, population_size, gene_length, max_generations, mutation_rate, gama, max_score, min_score):
        self.population_size = population_size
        self.gene_length = gene_length
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.gama = gama
        self.dis_max = max_score
        self.dis_min = min_score

    def resource_distribution_score(self, remaining_resource_nodes):
        scores = {
            key: self.dis_min if value <= 0 else (self.dis_max if value == resource_of_each_node else (value / resource_of_each_node) * self.dis_max)
            for key, value in remaining_resource_nodes.items()
        }
        return sum(scores.values())

    def check_nodes_remaining_resource(self, gene, original_nodes, resource_usage):
        nodes = copy.deepcopy(original_nodes)
        for i, j in enumerate(gene):
            nodes[j] -= resource_usage[i]
        return any(value < 0 for value in nodes.values())

    def initialize_population(self, nodes, usage):
        filtered_nodes = [k for k, v in nodes.items() if v > 0]
        if len(filtered_nodes) < len(usage):
            return -100

        population_size_in_fact = min(math.perm(len(filtered_nodes), self.gene_length), self.population_size)
        while True:
            population = [random.sample(filtered_nodes, self.gene_length) for _ in range(population_size_in_fact)]
            if any(not self.check_nodes_remaining_resource(gene, nodes, usage) for gene in population):
                return population

    def fitness(self, gene, usage, nodes, bandwidth, flow_size):
        original_nodes = copy.deepcopy(nodes)
        for j, i in enumerate(gene):
            original_nodes[i] -= usage[j]
        if any(value < 0 for value in original_nodes.values()):
            return -1e10 
        socre_dis = self.resource_distribution_score(original_nodes)
        total_latency = calc_latency(gene, bandwidth, flow_size)
        return (self.gama * socre_dis) - ((1 - self.gama) * total_latency)

    def selection(self, population, usage, nodes, bandwidth, flow_size):
        weights = [self.fitness(gene, usage, nodes, bandwidth, flow_size) for gene in population]
        min_weight = min(weights)
        adjusted_weights = [w + abs(min_weight) + 1 if min_weight <= 0 else w for w in weights]
        parent1 = random.choices(population, weights=adjusted_weights, k=1)[0]
        remaining_population = [ind for ind in population if ind != parent1]
        remaining_weights = [adjusted_weights[i] for i in range(len(population)) if population[i] != parent1]
        if all(x == 0 for x in remaining_weights):
            return parent1, [-1, -1, -1]
        parent2 = random.choices(remaining_population, weights=remaining_weights, k=1)[0]
        return parent1, parent2

    def crossover(self, parent1, parent2):
        point = random.randint(0, self.gene_length - 1)
        child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        if len(set(child1)) == len(child1) and len(set(child2)) == len(child2):
            return child1, child2
        return self.crossover(parent1, parent2)

    def mutate(self, gene, nodes):
        mutation_indices = [i for i in range(self.gene_length) if random.random() < self.mutation_rate]
        mutate_choice = [node for i, node in enumerate(gene) if i not in mutation_indices]
        if not mutation_indices:
            return gene

        for i in mutation_indices:
            gene[i] = random.choice([node for node in nodes.keys() if node not in gene])

        return gene

    def run(self, nodes, usage, bandwidth, flow_size):
        population = self.initialize_population(nodes, usage)
        if population == -100:
            return [-1, -1, -1], -1e10  # 返回一个表示错误的gene

        for generation in range(self.max_generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.selection(population, usage, nodes, bandwidth, flow_size)
                if parent2 == [-1, -1, -1]:
                    return parent1, self.fitness(parent1, usage, nodes, bandwidth, flow_size)

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, nodes)
                child2 = self.mutate(child2, nodes)

                new_population.extend([child1, child2])

            population = [list(individual) for individual in set(tuple(ind) for ind in new_population)]
            best_gene = max(population, key=lambda gene: self.fitness(gene, usage, nodes, bandwidth, flow_size))
            print(f'Generation {generation+1}: Best Gene = {best_gene}, Fitness = {self.fitness(best_gene, usage, nodes, bandwidth, flow_size)}')

        best_gene = max(population, key=lambda gene: self.fitness(gene, usage, nodes, bandwidth, flow_size))
        return best_gene, self.fitness(best_gene, usage, nodes, bandwidth, flow_size)

#---------------------------------------遗传算法----------------------------------------------#

GACADA = GeneticAlgorithmCada(pop_size, num_virtual_services, max_gen, mutation_rate, gama, score_max, score_min)
queue_data = pd.DataFrame(columns=[REQ_NO, 'Size', 'Waiting_Time', 'Resource_Usage'])
req_data = pd.DataFrame(columns=[REQ_NO, 'Nodes', 'Fitness', 'Arrival_Time', 'Start_Time', 'End_Time', 'Queue_Time', 'Delay_Time', 'Response_Time', 'Resource_Usage', 'RTP_node', 'RTP_node_avg', 'Deployed'])
time_data = pd.DataFrame(columns=['Time', 'Processing_Reqs', 'Num_of_Processing_Reqs', 'Processed_Reqs', 'Num_of_Processed_Reqs', 'Queuing_Reqs', 'Num_of_Queuing_Reqs', 'Refused_Reqs', 'Num_of_Refused_Reqs', 'Acceptance_Rate', 'RTP_time'])
Num_of_Demands = 1

for t in range(0, input_data['Time'].iloc[-1] + 1, sample_frequency):
    # 先对正在处理的需求进行判断，如果处理完毕，则释放需求所占用的资源
    time_data.loc[len(time_data), 'Time'] = t
    for index, row in req_data.iterrows():
        if row['Deployed'] == 1:
            continue
        if row['End_Time'] <= t and not pd.isna(row['Start_Time']):
            req_data.loc[index, 'Queue_Time'] = row['Start_Time'] - row['Arrival_Time']
            req_data.loc[index, 'Response_Time'] = row['End_Time'] - row['Arrival_Time']
            req_data.loc[index, 'Deployed'] = 1
            release_resource(row['Nodes'], row['Resource_Usage'], virtual_nodes)
            continue
        if row['Delay_Time'] < t and pd.isna(row['Start_Time']):
            req_data.loc[index, 'Deployed'] = -1
            queue_data.drop(queue_data[queue_data[REQ_NO] == row[REQ_NO]].index, inplace=True)

    # 判断这个时刻是否有新的需求到来
    if input_data['Time'].eq(t).any():
        queue_data.reset_index(drop=True, inplace=True)
        Match_Row = input_data[input_data['Time'] == t].iloc[0]
        Match_Row['Size'] = json.loads(Match_Row['Size'])
        Match_Row['Delay'] = json.loads(Match_Row['Delay'])
        Match_Row['Resource'] = json.loads(Match_Row['Resource'])
        for i in range(Match_Row['Arrival Demand']):
            queue_data.loc[len(queue_data)] = [
                int(Num_of_Demands),
                Match_Row['Size'][i],
                Match_Row['Delay'][i],
                Match_Row['Resource'][i]
            ]
            req_data.loc[len(req_data), [REQ_NO, 'Arrival_Time', 'Delay_Time', 'Resource_Usage']] = [Num_of_Demands, t, t + Match_Row['Delay'][i], Match_Row['Resource'][i]]
            Num_of_Demands += 1

    # 开始进行处理
    if queue_data.empty:
        print(f'在t={t}时刻,当前排队队列为空')
        continue

    req_to_drop = []
    for index, row in queue_data.iterrows():
        skip_flag = check_current_nodes_suit_for_current_req(virtual_nodes, row['Resource_Usage'])
        if not skip_flag:
            logger.debug(f"需求编号{row[REQ_NO]}的需求无法部署！")
            continue

        best_x, best_fitness = GACADA.run(virtual_nodes, row['Resource_Usage'], symmetric_bandwidth_data, row['Size'])

        if best_x == [-1, -1, -1]:
            logger.debug(f"需求编号{row[REQ_NO]}的需求无法部署！")
            continue

        original_nodes = copy.deepcopy(virtual_nodes)
        for j, i in enumerate(best_x):
            original_nodes[i] -= row['Resource_Usage'][j]
        if any(value < 0 for value in original_nodes.values()):
            logger.debug(f"需求编号{row[REQ_NO]}的需求无法部署！")
        else:
            Duration_Time = calc_latency(best_x, symmetric_bandwidth_data, row['Size'])
            req_data.loc[req_data[REQ_NO] == int(row.iloc[0]), 'Nodes'] = str(best_x)
            req_data.loc[req_data[REQ_NO] == int(row.iloc[0]), ['Start_Time', 'End_Time']] = [t, t + Duration_Time]
            req_data.loc[req_data[REQ_NO] == int(row.iloc[0]), 'Fitness'] = best_fitness
            RTP_values = [resource * Duration_Time for resource in row['Resource_Usage']]
            req_data.loc[req_data[REQ_NO] == int(row.iloc[0]), 'RTP_node'] = [RTP_values]
            req_data.loc[req_data[REQ_NO] == int(row.iloc[0]), 'RTP_node_avg'] = sum(RTP_values) / len(RTP_values)
            for j, k in enumerate(best_x):
                virtual_nodes[k] -= row['Resource_Usage'][j]
            req_to_drop.append(index)

    queue_data.drop(req_to_drop, inplace=True)
    processing_req = req_data[(req_data['Start_Time'].notna()) & (req_data['Deployed'].isna())][REQ_NO].tolist()
    processed_req = req_data[(req_data['Deployed'] == 1)][REQ_NO].tolist()
    refused_req = req_data[(req_data['Deployed'] == -1)][REQ_NO].tolist()
    queue_req = req_data[(req_data['Start_Time'].isna()) & (req_data['Deployed'].isna())][REQ_NO].tolist()

    acceptance_rate = len(processing_req + processed_req) / max(1, len(processing_req + processed_req + refused_req + queue_req))

    time_data.loc[time_data['Time'] == t, ['Processing_Reqs', 'Num_of_Processing_Reqs', 'Processed_Reqs', 'Num_of_Processed_Reqs', 'Queuing_Reqs', 'Num_of_Queuing_Reqs', 'Refused_Reqs', 'Num_of_Refused_Reqs', 'Acceptance_Rate']] = \
        [str(processing_req), len(processing_req), str(processed_req), len(processed_req), str(queue_req), len(queue_req), str(refused_req), len(refused_req), acceptance_rate]

    RTP_sum = sum(req_data.loc[req_data[REQ_NO].isin(processing_req), 'RTP_node_avg'])
    time_data.loc[time_data['Time'] == t, 'RTP_time'] = RTP_sum / max(1, len(processing_req))

# 如果到达结束时间还有需求在队列，则直接丢弃；还在处理的认为deployed
req_data.loc[(req_data['Start_Time'].notna()) & (req_data['Deployed'].isna()), 'Deployed'] = 1
req_data['Queue_Time'] = req_data['Start_Time'] - req_data['Arrival_Time']
req_data['Response_Time'] = req_data['End_Time'] - req_data['Arrival_Time']

counts = req_data['Deployed'].value_counts()
count_1 = counts.get(1, 0)
count_neg_1 = counts.get(-1, 0)
logger.debug(f"有{count_1}个需求被部署，有{count_neg_1}个需求被舍弃！")

save_output_csv(req_data, "GA_data_ouput.csv")
save_output_csv(time_data, "GA_time_output.csv")