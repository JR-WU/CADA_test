import networkx as nx
import matplotlib.pyplot as plt

# 创建一个全连接图
G = nx.complete_graph(9)

# 绘制图
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)  # 选择布局
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')

# 显示图
plt.title("9-Node Complete Graph")
plt.show()
