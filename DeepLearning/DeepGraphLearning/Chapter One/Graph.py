import dgl
import torch as th
# 边 0->1, 0->2, 0->3, 1->3
u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))
print(g) # 图中节点的数量是DGL通过给定的图的边列表中最大的点ID推断所得出的
# 获取节点的ID
print(g.nodes())
# 获取边的对应端点
print(g.edges())
# 获取边的对应端点和边ID
print(g.edges(form='all'))
# # 如果具有最大ID的节点没有边，在创建图的时候，用户需要明确地指明节点的数量。
# g = dgl.graph((u, v), num_nodes=8)

bg = dgl.to_bidirected(g)
print(bg.edges())

edges = th.tensor([2, 5, 3]), th.tensor([3, 5, 0])
g64 = dgl.graph(edges)
print(g64.idtype)

g32 = dgl.graph(edges, idtype=th.int32)
print(g32.idtype)

'''
定义节点和边的特征
'''

g = dgl.graph((th.tensor([0, 0, 1, 5]), th.tensor([1, 2, 2, 0])), idtype=th.int32)
print(g)

print(th.ones(2, 3))

g.ndata['x'] = th.ones(g.num_nodes(), 3)
g.edata['x'] = th.ones(g.num_edges(), dtype=th.int32)
print(g)

g.ndata['y'] = th.randn(g.num_nodes(), 5)
# 特征张量使用行优先的云泽，每个行切片储存1个节点或1条边的特征。
print(g.ndata['x'][1])
print(g.edata['x'][th.tensor([0, 3])])

# 边 0->1, 0->2, 0->3, 1->3
edges = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
weights = th.tensor([0.1, 0.6, 0.9, 0.7])  # 每条边的权重
g = dgl.graph(edges)
g.edata['w'] = weights  # 将其命名为 'w'
print(g)

'''
通过外部源创建图
'''

import scipy.sparse as sp

spmat = sp.rand(100, 100, density=0.05) # 5%非零项
print(dgl.from_scipy(spmat))
# 来自SciPy


import networkx as nx
nx_g = nx.path_graph(5) # 一条链路0-1-2-3-4
print(dgl.from_networkx(nx_g)) # 来自NetworkX

nxg = nx.DiGraph([(2, 1), (1, 2), (2, 3), (0, 0)])
print(dgl.from_networkx(nxg))

'''
异构图
'''

import dgl
import torch as th
# 创建一个具有3种节点类型和3种边类型的异构图
graph_data = {
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
   ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
}
g = dgl.heterograph(graph_data)
print(g.ntypes)
print(g.etypes)
print(g.canonical_etypes)
print(g)

# g = dgl.heterograph({
#    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
#    ('drug', 'is similar', 'drug'): (th.tensor([0, 1]), th.tensor([2, 3]))
# })
# print(g.nodes())
# # 设置/获取单一类型的节点或边特征，不必使用新的语法
# g.ndata['hv'] = th.ones(4, 1)

eg = dgl.edge_type_subgraph(g, [('drug', 'interacts', 'drug'), ('drug', 'treats', 'disease')])
print(eg)

'''
转换为同构图
'''

g = dgl.heterograph({
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))})
g.nodes['drug'].data['hv'] = th.zeros(3, 1)
g.nodes['disease'].data['hv'] = th.ones(3, 1)
g.edges['interacts'].data['he'] = th.zeros(2, 1)
g.edges['treats'].data['he'] = th.zeros(1, 2)
# 默认情况下不进行特征合并
hg = dgl.to_homogeneous(g)
print('hv' in hg.ndata)

# # 拷贝边的特征
# # 对于要拷贝的特征，DGL假定不同类型的节点或边的需要合并的特征具有相同的大小和数据类型
# hg = dgl.to_homogeneous(g, edata=['he'])
# # 拷贝节点特征
# hg = dgl.to_homogeneous(g, ndata=['hv'])
# print(hg.ndata['hv'])

'''
在GPU上训练
'''

import dgl
import torch as th
u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
g = dgl.graph((u, v))
g.ndata['x'] = th.randn(5, 3)   # 原始特征在CPU上
print(g.device)


cuda_g = g.to('cuda:0')         # 接受来自后端框架的任何设备对象
print(cuda_g.device)

print(cuda_g.ndata['x'].device )       # 特征数据也拷贝到了GPU上
# 由GPU张量构造的图也在GPU上
u, v = u.to('cuda:0'), v.to('cuda:0')
g = dgl.graph((u, v))
print(g.device)