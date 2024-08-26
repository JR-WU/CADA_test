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
debug_mode = False  # 设置为True以启用调试模式！！！！！！
if debug_mode:
    logger.setLevel(logging.DEBUG)

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

def save_output_csv(output_csv, output_name):
    # 获取当前脚本所在的绝对路径
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # 获取上一级目录
    parent_directory = os.path.dirname(current_directory)
    
    # 定义要检查的目录名称
    data_folder_path = os.path.join(parent_directory, 'data')
    
    # 检查data文件夹是否存在
    if not os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        raise FileNotFoundError("data文件夹不存在,请去上一级目录新建一个data文件夹,并将输入数据放入其中！")

    #存储csv文件
    csv_path = os.path.join(data_folder_path, output_name)
    output_csv.to_csv(csv_path, index=True)

def create_cada_nodes(NumOfHardNet, NumOfStep, resource):
    #计算字典的大小
    num_keys = NumOfHardNet * NumOfStep
    #创建字典，每个key对应的值都是resource
    cada_nodes = {i: resource for i in range(num_keys)}
    return cada_nodes

def create_resource_usage(num_ser,res_use):
    if len(res_use) != num_ser:
        raise ValueError("输入的资源使用数组长度与定义的虚拟化服务个数不匹配！")
    else:
        return res_use
    
def duplicates_check(lst):
    return len(lst) != len(set(lst)) #True有重复元素，False没有重复元素

def release_resource(req_nodes, resource_usage, nodes):
    req_nodes = json.loads(req_nodes)
    for j, k in enumerate(req_nodes):
        nodes[k] = nodes[k] + resource_usage[j]
        nodes[k] = round(nodes[k],2)

def cal_remaining_resource(nodes_resource,gene,resource_usage):
    for j, i in enumerate(gene):
        nodes_resource[i] = nodes_resource[i] - resource_usage[j]
        nodes_resource[i] = round(nodes_resource[i],2)

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

def cal_lantency(gene, bandwidth,flow_size):
    # 计算传输时延
    total_latency = 0
    for m in range(len(gene) - 1):
        # 获取链路的起点和终点
        start_node = gene[m]
        end_node = gene[m + 1]
        # 查找链路的带宽
        if (start_node, end_node) in bandwidth:
            band = bandwidth[(start_node, end_node)]
        elif (end_node, start_node) in bandwidth:
            band = bandwidth[(end_node, start_node)]
        else:
            raise ValueError(f"在计算持续时间时，链路 ({start_node}, {end_node}) 没有定义带宽。")
        # 计算链路的传输时延
        latency = flow_size / band
        total_latency += latency
    total_latency = round(total_latency,2)
    return total_latency

def check_current_nodes_suit_for_current_req(nodes, usage):
    # 复制节点资源，避免直接修改原始数据
    temp_nodes = list(nodes.values())
    usage_backup = usage.copy()
    # 按照降序排列 usage，这样我们先满足最大的需求
    usage_backup.sort(reverse=True)

    can_satisfy = True

    # 逐个需求检查能否分配给任意节点
    for u in usage_backup:
        possible = False
        for i in range(len(temp_nodes)):
            if temp_nodes[i] >= u:
                temp_nodes[i] -= u
                temp_nodes.pop(i)
                possible = True
                break  # 找到合适的节点后，立即跳出循环
        if not possible:
            can_satisfy = False
            break
    return can_satisfy


#环境参数初始化
num_virtual_services = 3
Number_of_Hardware_Networks = 3 #硬件网络个数
resource_of_each_node = 5
sample_frequency = 1 #采样频率，默认为1s，最小为0.01效果最好，但肯定最慢
REQ_NO = 'Request_Number'
score_min = 0
score_max = 1.5

#遗传算法参数
pop_size = 30 #种群数量
max_gen = 10 #迭代次数
mutation_rate = 0.01 #突变率
gama = 0.5 #适应度函数中资源利用率的权重 1-0.8为传输延迟权重

#读取、创建实验需要的数据
input_data = read_input_csv('generated_data.csv', True) # 读取输入数据
virtual_nodes = create_cada_nodes(Number_of_Hardware_Networks,num_virtual_services,resource_of_each_node) #设定虚拟网络节点数和每个节点的资源
bandwidth_pair_nodes = check_networking_setting('settings.txt', num_virtual_services, Number_of_Hardware_Networks)
#生成对称数据,symmetric为全连接网络中的带宽
symmetric_bandwidth_data = bandwidth_pair_nodes.copy()
for (i,j), value in bandwidth_pair_nodes.items():
    symmetric_bandwidth_data[j,i] = value

#遗传算法求解资源分配问题
class GeneticAlgorithmCada:
    def __init__(self, population_size, gene_length, max_generations, mutation_rate, gama, max_score, min_score):
        self.population_size = population_size
        self.gene_length = gene_length
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.gama = gama
        self.dis_max = max_score
        self.dis_min = min_score

    
    def resource_distribution_score(self,remaining_resource_nodes):
        scores = {}
        for key, value in remaining_resource_nodes.items():
            if value <= 0:
                scores[key] = self.dis_min
            elif value == resource_of_each_node:
                scores[key] = self.dis_max
            else:
                scores[key] = (value/resource_of_each_node) * self.dis_max
        total_sum = sum(scores.values())
        return total_sum

    def check_nodes_remaining_resource(self, gene, original_nodes, resource_usage):#判断采用某个gene后，节点资源是否小于0
        nodes = copy.deepcopy(original_nodes)
        j = 0
        for i in gene:
            nodes[i] = nodes[i] - resource_usage[j]
            j = j + 1
        count = sum(1 for value in nodes.values() if value < 0)
        if count > 0 :
            return True
        else:
            return False


    def initialize_population(self, nodes, usage):
        filtered_nodes = {k: v for k, v in nodes.items() if v > 0} #去掉资源为小于等于0的节点，以提高运行效率
        node_list = list(filtered_nodes.keys())

        #判断节点数量是否支撑该需求的分配
        if len(node_list) < len(usage):
            logger.debug(f"可用节点数量:{len(node_list)}已经不足以进行部署")
            return -100

        #种群最大取P(9,3),但如果种群大于population_size，则取population_size，否则就按P(total,want)的形式计算出种群
        population_size_in_fact = math.perm(len(node_list), self.gene_length) 
        if population_size_in_fact > self.population_size:
            population_size_in_fact = self.population_size

        #构建种群
        while True:
            population = []
            count = 0
            while len(population) < population_size_in_fact:
                gene = random.sample(node_list, self.gene_length)
                if gene not in population:
                    population.append(gene)

            #判断种群是否存在至少一个gene，在分配该gene到节点上后，分配的节点剩余资源大于等于0
            for gene in population:
                if self.check_nodes_remaining_resource(gene, nodes, usage):
                    count += 1
            if count != len(population):
                return population

    def fitness(self, gene, usage, nodes, bandwidth, flow_size):
        original_nodes_fitness = copy.deepcopy(nodes)
        #计算资源分布分数
        cal_remaining_resource(original_nodes_fitness, gene, usage)
        # for j, i in enumerate(gene):
        #     original_nodes_fitness[i] -= usage[j]
        has_negative = any(value < 0 for value in original_nodes_fitness.values())
        if has_negative:# 如果资源变为负数，返回一个非常低的适应度值，以保证遗传算法不会选这个gene
            return -1e10 
        score_nodes = copy.deepcopy(original_nodes_fitness)
        socre_dis = self.resource_distribution_score(score_nodes)

        # 计算传输时延
        total_latency = 0
        for m in range(len(gene) - 1):
            # 获取链路的起点和终点
            start_node = gene[m]
            end_node = gene[m + 1]
            # 查找链路的带宽
            if (start_node, end_node) in bandwidth:
                band = bandwidth[(start_node, end_node)]
            elif (end_node, start_node) in bandwidth:
                band = bandwidth[(end_node, start_node)]
            else:
                raise ValueError(f"链路 ({start_node}, {end_node}) 没有定义带宽。")
            # 计算链路的传输时延
            latency = flow_size / band
            total_latency += latency
        fitness_value = ((self.gama * socre_dis) + ((1- self.gama) * -total_latency))
        fitness_value = round(fitness_value, 2) #取2位小数
        # if fitness_value <= 0: #如果适应度值小于等于0，说明分数少或者延迟高，那可以认为这个选择不咋地。但也比node的资源为负数强。
        #     return 2e-6
        return fitness_value

    def selection(self, population, usage, nodes, bandwidth, flow_size):
        weights = []
        for gene in population:
            original_nodes = copy.deepcopy(nodes)  # 在每次计算前重置 nodes
            weight = self.fitness(gene, usage, original_nodes, bandwidth, flow_size)
            weights.append(weight)
        #因为fitness有负数，因此我们采用偏移权重,即给每个权重加100保证均为正数
        min_weight = min(weights)
        # 如果最小权重是负数，则将所有权重加上一个偏移量
        if min_weight <= 0:
            offset = abs(min_weight) + 1
            adjusted_weights = [w + offset for w in weights]
            logger.debug(f"由于存在一个或者多个权重小于等于0，因此为每个权重添加偏移量{offset},偏移后的权重为{adjusted_weights}")
        else:
            adjusted_weights = weights
        # 第一次选择
        parent1 = random.choices(population, weights=adjusted_weights, k=1)[0]
        # 从 population 中去除 parent1
        remaining_population = [ind for ind in population if ind != parent1]
        remaining_weights = [adjusted_weights[i] for i in range(len(population)) if population[i] != parent1]
        # 第二次选择
        if all(x == 0 for x in remaining_weights):
            logger.debug("列表中的所有元素都是0")
            parent2 = [-1,-1,-1,-1]
            return parent1, parent2
        parent2 = random.choices(remaining_population, weights=remaining_weights, k=1)[0]
        return parent1, parent2

    def crossover(self, parent1, parent2):
        n = 1
        while True:
            logger.debug(f"开始选择交叉选择,第{n}次")
            point = random.randint(0, self.gene_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]

            # 交叉可能会出现child中有两个相同节点的情况，因此如果遇到这种情况就要重新来。
            # 检查子代基因是否有重复元素

            if len(set(child1)) == len(child1) and len(set(child2)) == len(child2):
                return child1, child2
            else:
                logger.debug(f"Childs中存在相同节点, 其中child1:{child1}, child2: {child2}")
                n = n + 1
                child1 = []
                child2 = []

#变异过程：先为gene中的每个元素计算随机生成其突变概率，找出小于突变率的元素的索引。之后，对每个需要突变的元素，其变化的范围有以下约束：1、不能突变成自己，2、不能突变成不会变异的元素
    def mutate(self, gene, nodes):
        logger.debug("开始突变!")
        rate_for_each_node_in_gene = [random.random() for _ in range(self.gene_length)]
        mutation_indices = [i for i, v in enumerate(rate_for_each_node_in_gene) if v < self.mutation_rate]
        not_mutation_indices = [i for i, v in enumerate(rate_for_each_node_in_gene) if v > self.mutation_rate]
        mutate_choice = list(nodes.keys())[:]

        #检查是否存在变异的基因，存在继续，不存在直接返回
        if len(mutation_indices) == 0:
            logger.debug(f"child:{gene}没有发生突变")
            return gene
        
        logger.debug(f"child:{gene}发生突变")
        #执行突变，先去除不会突变的节点，然后随机选择突变节点
        for index in not_mutation_indices:
                mutate_choice.remove(gene[index])
        for i in mutation_indices:
            if gene[i] in mutate_choice:
                mutate_choice.remove(gene[i])
            backup = gene[i]
            gene[i] = random.choice(mutate_choice)
            if backup not in gene:
                mutate_choice.append(backup)
            mutate_choice.remove(gene[i])
        logger.debug(f"突变为child:{gene}")
        return gene

    def run(self, nodes, usage, bandwidth, flow_size):
        population = self.initialize_population(nodes,usage)
        if population == -100: #如果返回的是一个数为-100则表示网络中已经无法支撑后续节点的部署，-100也为本实验的错误代码
            fitness_best = -1e10
            return [-1,-1,-1],fitness_best #返回一个表示错误的gene
        for generation in range(self.max_generations):
            new_population = []
            #加一个判断，如果population小于等于1，
            for _ in range(self.population_size // 2):
                logger.debug("Step1:开始选择parents")
                logger.debug(f"population:{population}")
                parent1, parent2 = self.selection(population, usage, nodes, bandwidth, flow_size)
                logger.debug(f"选择的parent1: {parent1}, parent2: {parent2}")

                if set(parent1) != {-1} and set(parent2) == {-1}:
                    return parent1, self.fitness(parent1, usage, nodes, bandwidth, flow_size)

                if duplicates_check(parent1) or duplicates_check(parent2):
                    raise TypeError("父genes存在相同元素")

                logger.debug("Step2:生成childs")
                child1, child2 = self.crossover(parent1, parent2)
                logger.debug(f"child1: {child1}, child2: {child2}")

                logger.debug("Step3:变异")
                child1 = self.mutate(child1,nodes)
                child2 = self.mutate(child2,nodes)
                logger.debug(f"mutated_child1: {child1}, mutated_child2: {child2}")

                if duplicates_check(child1) or duplicates_check(child2):
                    raise TypeError("childs存在相同元素")
                new_population.extend([child1, child2])
            # unique_population = list(set(tuple(individual) for individual in new_population))
            # unique_population = [list(individual) for individual in unique_population] #去重
            population = [list(individual) for individual in set(tuple(individual) for individual in new_population)]
            best_gene = max(population, key=lambda gene: self.fitness(gene, usage, nodes, bandwidth, flow_size))
            print(f'Generation {generation+1}: Best Gene = {best_gene}, Fitness = {self.fitness(best_gene, usage, nodes, bandwidth, flow_size)}')
        best_gene = max(population, key=lambda gene: self.fitness(gene, usage, nodes, bandwidth, flow_size))
        best_x = best_gene
        return best_x, self.fitness(best_gene, usage, nodes, bandwidth, flow_size)
    
#---------------------------------------遗传算法----------------------------------------------#

GACADA = GeneticAlgorithmCada(pop_size, num_virtual_services, max_gen, mutation_rate, gama, score_max, score_min)
#新建一个dataframe,来表示排队序列, 该df有四列, 分别为需求名、需求大小、最大等待时间、资源占用情况
queue_data = pd.DataFrame(columns=[REQ_NO, 'Size', 'Waiting_Time','Resource_Usage'])
#再次新建一个dataframe, 来表示每个需求的结果，包括需求名、分配到的节点、适应度、到达时间、开始处理时间、完成处理时间、排队时间、响应时间、资源占用情况、资源时间积、部署是否成功（这里的部署是否成功指的是该需求已经处理完毕）。
#若需求处理失败，则除了需求名和部署是否成功其余为NaN
req_data = pd.DataFrame(columns=[REQ_NO, 'Nodes','Fitness', 'Arrival_Time', 'Start_Time', 'End_Time', 'Queue_Time', 'Delay_Time', 'Response_Time', 'Resource_Usage','RTP_node', 'RTP_node_avg', 'Deployed'])
#根据时间创建的csv,用来查看接收率和资源时间积等
time_data = pd.DataFrame(columns=['Time', 'Processing_Reqs', 'Num_of_Processing_Reqs', \
                                'Processed_Reqs','Num_of_Processed_Reqs','Queuing_Reqs', \
                                'Num_of_Queuing_Reqs','Refused_Reqs','Num_of_Refused_Reqs', \
                                'Acceptance_Rate','RTP_time'])
Num_of_Demands = 1

#以时间来循环, 步长为1;步长即决策大脑每隔1s检查网络需求的到达情况，以及排队序列中的需求情况；步长会影响相应时间和资源占用情况，步长越短，响应时间小，资源占用情况越理想。
for t in range(0,input_data['Time'].iloc[-1] + 1, sample_frequency):
    #先对正在处理的需求进行判断，如果处理完毕，则释放需求所占用的资源
    logger.debug(f"当t={t}时，在分配需求前，当前网络节点的资源情况是{virtual_nodes}")
    time_data.loc[len(time_data), ['Time']] = t #为time_data数据添加时刻
    for index, row in req_data.iterrows():
        if row['Deployed'] == 1: #如果已经成功部署，则不用再去遍历该req
            continue
        if row['End_Time'] <= t and not pd.isna(row['Start_Time']): #存在开始时间同时结束时间小于当前时刻，表示该需求已经被处理完毕
            req_data.loc[index, 'Queue_Time'] = row['Start_Time'] - row['Arrival_Time']
            req_data.loc[index, 'Response_Time'] = row['End_Time'] - row['Arrival_Time']
            req_data.loc[index, 'Deployed'] = 1 
            #释放占用资源
            release_resource(row['Nodes'], row['Resource_Usage'], virtual_nodes)
            continue
        #如果需求的时延已经大于当前时间，则该需求在queue中被舍弃，在req_data中表明为false
        if row['Delay_Time'] < t and pd.isna(row['Start_Time']): #不存在开始时间且最晚执行时间已经小于当前时刻，则认为其被丢弃！
            req_data.loc[index, 'Deployed'] = -1
            index_to_drop = queue_data[queue_data[REQ_NO] == row[REQ_NO]].index
            queue_data = queue_data.drop(index_to_drop)

    #判断这个时刻是否有新的需求到来
    if input_data['Time'].eq(t).any():
        queue_data.reset_index(drop=True, inplace=True) #重置索引
        #将新到来的需求添加值queue_data中
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
            req_data.loc[len(req_data), [REQ_NO,'Arrival_Time','Delay_Time','Resource_Usage']] = [Num_of_Demands, t, t + Match_Row['Delay'][i], Match_Row['Resource'][i]]
            Num_of_Demands = Num_of_Demands + 1

    logger.debug(f"t={t},对当前Queue中的数据进行遗传算法")
    req_to_drop = [] #用来标记执行后要从队列删除的需求
    for index, row in queue_data.iterrows():
        print(f"t={t}时刻,处理编号为{row[REQ_NO]}的需求，此时节点状态为{virtual_nodes}")
        #判断此时各个节点的资源是否能满足该需求，若资源已经无法满足，则该需求连GA都不用跑了，直接去等着下一时刻
        skip_flag = check_current_nodes_suit_for_current_req(virtual_nodes, row['Resource_Usage'])
        if not skip_flag:
            print(f"当t={t}时，当前网络节点的资源情况是{virtual_nodes},此时网络因为节点资源已经无法给当前需求提供至少一种分配方案，故编号{row[REQ_NO]}的需求，其资源占用情况是{row['Resource_Usage']}，无法部署！")
            continue

        #开始GA
        best_x, best_fitness = GACADA.run(virtual_nodes, row['Resource_Usage'], symmetric_bandwidth_data, row['Size'])#遗传算法找最优
        print(f"在t={t}时刻，为需求{row[REQ_NO]}分配的链路为{best_x}，适应度为{best_fitness}")
        #对结果进行判断
        if best_x == [-1,-1,-1]:
            print(f"当t={t}时，当前网络节点的资源情况是{virtual_nodes},此时网络因为节点数量不够，已经无法满足需求编号{row[REQ_NO]}的需求进行部署！")
            continue
        original_nodes = copy.deepcopy(virtual_nodes)
        cal_remaining_resource(original_nodes,best_x,row['Resource_Usage'])
        # for j, i in enumerate(best_x):
        #     original_nodes[i] = original_nodes[i] - row['Resource_Usage'][j]
        #     original_nodes[i] = round(original_nodes[i],2) 
        if any(value < 0 for value in original_nodes.values()):
            print(f"当t={t}时，当前网络节点的资源情况是{virtual_nodes},此时网络因为资源数量不够，已经无法满足需求编号{row[REQ_NO]}的需求进行部署！")
            #break
        else:
            Duration_Time = cal_lantency(best_x, symmetric_bandwidth_data, row['Size'])
            req_data.loc[req_data[REQ_NO] == int(row.iloc[0]), 'Nodes'] = str(best_x)
            req_data.loc[req_data[REQ_NO] == int(row.iloc[0]), ['Start_Time','End_Time']] = [t, t + Duration_Time]
            req_data.loc[req_data[REQ_NO] == int(row.iloc[0]), ['Fitness']] = best_fitness
            RTP_values = [round(resource * Duration_Time,2) for resource in row['Resource_Usage']]
            req_data.loc[req_data[REQ_NO] == int(row.iloc[0]), 'RTP_node'] = [RTP_values]
            req_data.loc[req_data[REQ_NO] == int(row.iloc[0]), ['RTP_node_avg']] = round(sum(RTP_values)/len(RTP_values),2)
            cal_remaining_resource(virtual_nodes,best_x,row['Resource_Usage'])
            # for j, k in enumerate(best_x):
            #     virtual_nodes[k] = virtual_nodes[k] - row['Resource_Usage'][j]
            req_to_drop.append(index)
    queue_data = queue_data.drop(req_to_drop) #删除已经执行的需求
    logger.debug(f"当t={t}时，网络资源分配完毕，当前网络节点的资源情况是{virtual_nodes}，有{len(req_to_drop)}个需求被分配，有{queue_data.shape[0]}个需求还在队列中！")

    processing_req = []
    processed_req = []
    refused_req = []
    queue_req = []
    num_processing_req = 0 #正在处理的req数量
    num_processed_req = 0 #处理完毕的req数量
    num_refused_req = 0 #被拒绝的req数量
    num_queue_req = 0 #正在排队的req数量

    for index, row in req_data.iterrows():
        if not pd.isna(row['Start_Time']) and pd.isna(row['Deployed']):
            num_processing_req += 1
            processing_req.append(row[REQ_NO])
        elif not pd.isna(row['Start_Time']) and row['Deployed'] == 1:
            num_processed_req += 1
            processed_req.append(row[REQ_NO])
        elif pd.isna(row['Start_Time']) and row['Deployed'] == -1:
            num_refused_req += 1
            refused_req.append(row[REQ_NO])
        elif pd.isna(row['Start_Time']) and pd.isna(row['Deployed']):
            num_queue_req += 1
            queue_req.append(row[REQ_NO])
    
    acceptance_rate = (num_processing_req + num_processed_req)/ (num_processing_req + num_processed_req + num_refused_req + num_queue_req)

    time_data.loc[time_data['Time'] == t, ['Processing_Reqs', 'Num_of_Processing_Reqs', \
        'Processed_Reqs','Num_of_Processed_Reqs','Queuing_Reqs', \
        'Num_of_Queuing_Reqs','Refused_Reqs','Num_of_Refused_Reqs', 'Acceptance_Rate']] \
        = [str(processing_req), num_processing_req , str(processed_req), num_processed_req,\
        str(queue_req), num_queue_req, str(refused_req), num_refused_req, acceptance_rate]

    RTP_sum = 0
    for req in processing_req:
        RTP_sum += req_data.loc[req_data[REQ_NO] == req, 'RTP_node_avg'].values[0]
    time_data.loc[time_data['Time'] == t,['RTP_time']] = RTP_sum/len(processing_req) #每个时刻正在处理的需求的平均资源时间积


#如果到达结束时间还有需求在队列，则直接丢弃；还在处理的认为deployed。
for index, row in req_data.iterrows():
    if row.iloc[-1] == 1 or row.iloc[-1] == -1: #如果已经成功部署或者被抛弃，则不用再去遍历该req
        continue
    if not pd.isna(row['Start_Time']):
        req_data.loc[index, 'Deployed'] = 1 
        req_data.loc[index, 'Queue_Time'] = row['Start_Time'] - row['Arrival_Time']
        req_data.loc[index, 'Response_Time'] = row['End_Time'] - row['Arrival_Time']
    else:
        req_data.loc[index, 'Deployed'] = -1e5
rows_to_drop = req_data[req_data['Deployed'] == -1e5].index
req_data = req_data.drop(rows_to_drop)
counts = req_data['Deployed'].value_counts()
# 获取等于 1 和 -1 的数量
count_1 = counts.get(1, 0)  # 如果没有 1 的项，会返回 0
count_neg_1 = counts.get(-1, 0)  # 如果没有 -1 的项，会返回 0
logger.debug(f"有{count_1}个需求被部署，有{count_neg_1}个需求被舍弃！")
#存储GA算法的输出结果
save_output_csv(req_data,"GA_data_ouput.csv")
save_output_csv(time_data, "GA_time_output.csv")