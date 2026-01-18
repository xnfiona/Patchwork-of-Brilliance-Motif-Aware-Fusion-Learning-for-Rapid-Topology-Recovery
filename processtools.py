#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/26 3:18 下午
# @Site    : 
# @File    : processtools.py
# @Software: PyCharm
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx, to_networkx
import os
from collections import defaultdict
import math
import numpy as np
from collections import deque
import itertools
from itertools import chain
from itertools import islice
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


def is_triangle_motif(G, u, v):
    return len(set(nx.common_neighbors(G, u, v))) > 0


# 尝试在候选边中找到跨分量边用于连通图
def find_cross_component_edge(G, candidates):
    components = list(nx.connected_components(G))
    comp_map = {}
    for i, comp in enumerate(components):
        for node in comp:
            comp_map[node] = i
    for u, v in candidates:
        if comp_map.get(u) != comp_map.get(v):
            return (u, v)
    return None


# 构建 motif-aware 且共识度控制的超级图，边数精确限制
def consensus_motif_limited_supergraph(graph_list, target_edges=200):
    target_edges = len(graph_list[0].edges())
    edge_vote = defaultdict(int)
    motif_info = {}

    # 统计每条边的出现频率和 motif 状态
    for G in graph_list:
        for u, v in G.edges():
            key = tuple(sorted((u, v)))
            edge_vote[key] += 1
            if key not in motif_info:
                motif_info[key] = is_triangle_motif(G, u, v)
            else:
                motif_info[key] = motif_info[key] or is_triangle_motif(G, u, v)

    # 将边按投票等级和 motif 状态分类
    vote_bins = defaultdict(list)
    for edge, count in edge_vote.items():
        motif_flag = motif_info[edge]
        vote_bins[(count, motif_flag)].append(edge)

    # 排序规则：优先高投票 + 是 motif 边
    sorted_bins = sorted(vote_bins.keys(), key=lambda x: (x[0], x[1]), reverse=True)

    G_super = nx.Graph()
    all_nodes = set().union(*[G.nodes() for G in graph_list])
    G_super.add_nodes_from(all_nodes)

    # 逐级补边,需要保证拓扑结构是连通，且优秀的
    # for bin_key in sorted_bins:
    #     for edge in vote_bins[bin_key]:
    #         if len(G_super.edges()) < target_edges:
    #             G_super.add_edge(*edge)
    #         else:
    #             break
    #     if len(G_super.edges()) >= target_edges:
    #         break
    all_candidates = []
    for bin_key in sorted_bins:
        all_candidates.extend(vote_bins[bin_key])

    added_edges = set()
    for edge in all_candidates:
        if len(G_super.edges()) >= target_edges:
            break
        G_super.add_edge(*edge)
        added_edges.add(edge)

    # 如果不连通，则补上跨分量边直到连通
    while not nx.is_connected(G_super):
        remaining = [e for e in all_candidates if e not in added_edges]
        cross_edge = find_cross_component_edge(G_super, remaining)
        if cross_edge:
            G_super.add_edge(*cross_edge)
            added_edges.add(cross_edge)
        else:
            break  # 无法进一步连通

    # Step 2: 精确补边
    if len(G_super.edges()) < target_edges:
        remaining = [e for e in all_candidates if e not in added_edges]
        for e in remaining:
            if len(G_super.edges()) >= target_edges:
                break
            G_super.add_edge(*e)
            added_edges.add(e)

    # Step 3: 精确裁剪
    elif len(G_super.edges()) > target_edges:
        # 计算所有边的优先级（投票数 + motif标志）
        edge_priority = {}
        for edge in G_super.edges():
            key = tuple(sorted(edge))
            vote = edge_vote.get(key, 0)
            motif = motif_info.get(key, False)
            edge_priority[edge] = (vote, motif)

        # 按优先级升序排列（最低优先级的先删）
        sorted_edges = sorted(edge_priority.items(), key=lambda x: (x[1][0], x[1][1]))
        for edge, _ in sorted_edges:
            if len(G_super.edges()) <= target_edges:
                break
            G_super.remove_edge(*edge)
            if not nx.is_connected(G_super):  # 避免断图
                G_super.add_edge(*edge)  # 恢复
    return G_super


def uninon_graph(graph_list, target_edges=200):
    G_combined = nx.compose_all([one for one in graph_list])

    if len(G_combined.edges()) > target_edges:
        # 计算所有边的优先级（投票数 + motif标志）
        edge_priority = {}
        for edge in G_combined.edges():
            key = tuple(sorted(edge))
            vote = G_combined.get(key, 0)
            motif = motif_info.get(key, False)
            edge_priority[edge] = (vote, motif)

        # 按优先级升序排列（最低优先级的先删）
        sorted_edges = sorted(edge_priority.items(), key=lambda x: (x[1][0], x[1][1]))
        for edge, _ in sorted_edges:
            if len(G_combined.edges()) <= target_edges:
                break
            G_combined.remove_edge(*edge)
            if not nx.is_connected(G_combined):  # 避免断图
                G_combined.add_edge(*edge)  # 恢复

    # 创建新图，只保留top 200条边
    G_super = nx.Graph()
    for u, v, attr in top_edges:
        G_super.add_edge(u, v, **attr)

    # 设置画布
    fig, axs = plt.subplots(1, 5, figsize=(16, 5))

    # 画每个原始图
    titles = ['Graph 1', 'Graph 2', 'Graph 3', 'Graph 4','Combined Graph']
    for i, G in enumerate(graph_list):
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=axs[i], with_labels=True, node_color='lightblue', edge_color='gray')
        axs[i].set_title(titles[i])

    # 画合并后的图
    pos_combined = nx.spring_layout(G_super, seed=42)
    nx.draw(G_super, pos_combined, ax=axs[4], with_labels=True, node_color='lightgreen', edge_color='black')
    axs[4].set_title(titles[4])

    plt.tight_layout()
    plt.show()
    return G_super


def gcn_graph(G):
    degs = np.array([G.degree(n) for n in G.nodes()])  ## 可以换成motif覆盖度
    degs = torch.tensor(degs, dtype=torch.float32).view(-1, 1)
    # 改为 motif 覆盖度特征（如三角形数）
    triangles = nx.triangles(G)  # 返回 dict {node: num_triangles}
    motif_vec = np.array([triangles[n] for n in G.nodes()])
    features = np.array([[G.degree(n), triangles[n]] for n in G.nodes()])
    # x_feat = torch.tensor(motif_vec, dtype=torch.float32).view(-1, 1)
    # x_feat = (x_feat - x_feat.mean()) / (x_feat.std() + 1e-6)# 归一化
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    x_feat = torch.tensor(features, dtype=torch.float32)

    G_nx = G.copy()
    # 转换为 PyG 数据
    for i, n in enumerate(G_nx.nodes()):
        G_nx.nodes[n]['x'] = x_feat[i]
    data = from_networkx(G_nx)
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()
    return data


def calRM(G, G_start):
    N = len(G)
    if 'neighbor' not in G.nodes[0]:  # 检查缺项的，然后赋值
        for node in G_start.nodes:
            if 'neighbor' in G_start.nodes[node]:
                G.nodes[node]['neighbor'] = G_start.nodes[node]['neighbor']
    if 'loc' not in G.nodes[0]:  # 检查缺项的，然后赋值
        for node in G_start.nodes:
            if 'loc' in G_start.nodes[node]:
                G.nodes[node]['loc'] = G_start.nodes[node]['loc']
    loc_matrix = np.zeros((N, N))
    for i in range(N - 1):
        for j in range(i + 1, N):
            if G.nodes[i]['loc'] != [None, None] and G.nodes[j]['loc'] != [None, None]:
                dist = distance(G.nodes[i]['loc'], G.nodes[j]['loc'])
                if dist <= 200:
                    loc_matrix[i][j] = 1
    all_moti = [a for a in clique(G.copy(), loc_matrix) if len(a) == 3]

    moti_type = {0: 0, 1: 0, 2: 0, 3: 0}
    for moti in all_moti:
        t = Moti_3_Type(G, moti[0], moti[1], moti[2])
        moti_type[t] += 1
    s1 = moti_type[1] + moti_type[2] + moti_type[3]
    p1 = moti_type[1] / s1
    p2 = moti_type[2] / s1
    p3 = moti_type[3] / s1
    M = p1 * math.log(p1) + p2 * math.log(p2) + p3 * math.log(p3)
    return -M


def Moti_3_Type(G, i, j, k):
    sub = G.subgraph([i, j, k])
    m = len(sub.edges)
    if m == 0:
        return 0
    elif m == 1:
        return 1
    elif m == 2:
        return 2
    elif m == 3:
        return 3


def clique(G, loc_matrix):
    index = {}
    nbrs = {}
    n = len(G.nodes)
    for u in G:
        index[u] = len(index)
        nbrs[u] = [j for j in range(n) if loc_matrix[u][j] == 1]
    queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in G)
    while queue:
        base, cnbrs = map(list, queue.popleft())
        if len(base) == 4:
            break
        yield list(set(base))
        for i, u in enumerate(cnbrs):
            element = (chain(base, [u]),
                       list(set(islice(cnbrs, i + 1, None)) & set(G.nodes[u]['neighbor'])))
            queue.append(element)


def distance(loc1, loc2):
    return np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


def build_topk_graph(adj_matrix, top_k, original_degrees, G_start):
    # Connected Robust Graphs by Motif - guided Sampling
    N = adj_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node in G_start.nodes:  # 给他复制节点属性
        G.nodes[node]['loc'] = G_start.nodes[node]['loc']
        G.nodes[node]['neighbor'] = G_start.nodes[node]['neighbor']

    # Step 1: Ensure connectivity via MST
    full_G = nx.Graph()
    for i in range(N):
        for j in range(i + 1, N):
            if distance(G.nodes[i]['loc'], G.nodes[j]['loc']) <= 200:
                full_G.add_edge(i, j, weight=adj_matrix[i, j].item())
    mst = nx.maximum_spanning_tree(full_G)
    G.add_edges_from(mst.edges())

    current_degrees = [G.degree[i] for i in range(N)]
    current_edges = set(G.edges())
    remaining_edges = top_k - G.number_of_edges()

    # Step 2: Build fast-access map of triads
    edge_weights = {}
    for i in range(N):
        for j in range(i + 1, N):
            if distance(G.nodes[i]['loc'], G.nodes[j]['loc']) <= 200:
                edge_weights[(i, j)] = adj_matrix[i, j].item()

    triads = []
    for u, v, w in itertools.combinations(range(N), 3):
        score = edge_weights.get((min(u, v), max(u, v)), 0) + \
                edge_weights.get((min(u, w), max(u, w)), 0) + \
                edge_weights.get((min(v, w), max(v, w)), 0)
        triads.append(((u, v, w), score))
    triads.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Greedy sampling of motifs
    for (u, v, w), _ in triads:
        for (i, j) in [(u, v), (v, w), (u, w)]:
            if (i, j) in current_edges or (j, i) in current_edges:
                continue
            if not distance(G.nodes[i]['loc'], G.nodes[j]['loc']) <= 200:
                continue
            if current_degrees[i] >= original_degrees[i] or current_degrees[j] >= original_degrees[j]:
                continue
            G.add_edge(i, j)
            current_edges.add((i, j))
            current_degrees[i] += 1
            current_degrees[j] += 1
            remaining_edges -= 1
            if remaining_edges <= 0:
                break
        if remaining_edges <= 0:
            break

    # 如果边数不足，则补边
    while G.number_of_edges() < top_k:
        N = adj_matrix.shape[0]
        added = False
        # 所有剩余候选边（按权重排序）
        edges = [(i, j, adj_matrix[i, j].item())
                 for i in range(N) for j in range(i + 1, N)
                 if not G.has_edge(i, j) and distance(G.nodes[i]['loc'], G.nodes[j]['loc']) <= 200
                 or G.degree[i] < original_degrees[i] or G.degree[j] < original_degrees[j]]
        edges = sorted(edges, key=lambda x: x[2], reverse=True)
        for u, v, _ in edges:
            G.add_edge(u, v)
            added = True
            if G.number_of_edges() >= top_k:
                break
        if not added:
            break  # 无法再加边，终止
    return G





def graph_to_adj_tensor(G, max_N):
    A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
    N = A.shape[0]
    A_padded = torch.zeros(max_N, max_N)
    A_padded[:N, :N] = A
    return A_padded


class FlexibleGraphSequenceDataset(Dataset):
    def __init__(self, data, score_list, adj_true_list, sequence_length=1, gap_range=(1, 5), target_offset=1):
        """
        data: list of tensor embeddings at each time t (e.g., output of GCNEmbedder, shape: [64])
        input_len: how many time steps as input (e.g., 3)
        gap_range: range of random gaps between input steps (inclusive)
        target_offset: how far after the last input point to predict (e.g., t+1)
        """
        self.data = data
        self.score_list = score_list
        self.adj_true_list = adj_true_list
        self.seq_length = sequence_length
        self.gap_range = gap_range
        self.target_offset = target_offset
        self.samples = self.build_samples()

    def build_samples(self):
        samples = []
        L = len(self.data)
        for t_start in range(L):
            time_indices = [t_start]
            for _ in range(self.seq_length - 1):
                if time_indices[-1] + self.gap_range[1] >= L:
                    break
                gap = random.randint(*self.gap_range)
                next_t = time_indices[-1] + gap
                if next_t >= L:
                    break
                time_indices.append(next_t)

            if len(time_indices) == self.seq_length:
                t_target = time_indices[-1] + self.target_offset
                if t_target < L:
                    samples.append((time_indices, t_target))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        time_indices, t_target = self.samples[idx]
        # print("Index is:{}, {}".format(time_indices, t_target))
        x = [self.data[t] for t in time_indices]
        y = self.data[t_target]
        score_last = self.score_list[t_target-1]
        score = self.score_list[t_target]
        adj = self.adj_true_list[t_target]
        return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(y, dtype=torch.float32), score_last, score, adj


class ExhaustiveGraphSequenceDataset(Dataset):
    def __init__(self, data, score_list, adj_true_list, sequence_length=1, min_gap=1, target_offset=1):
        """
        graph_embeddings: list of tensor embeddings at each time t (e.g., output of GCNEmbedder, shape: [64])
        input_len: how many time steps as input (e.g., 3)
        target_offset: how far after the last input point to predict (e.g., t+1)
        min_gap: minimum gap between consecutive time steps to ensure time strictly increases
        """
        self.data = data
        self.score_list = score_list
        self.adj_true_list = adj_true_list
        self.seq_length = sequence_length
        self.min_gap = min_gap
        self.target_offset = target_offset
        self.samples = self.build_all_combinations()

    def build_all_combinations(self):
        L = len(self.data)
        samples = []
        for t_comb in itertools.combinations(range(L), self.seq_length):
            # ensure gaps between time steps are at least min_gap
            if all((t2 - t1 >= self.min_gap) for t1, t2 in zip(t_comb, t_comb[1:])):
                t_target = t_comb[-1] + self.target_offset
                if t_target < L:
                    samples.append((list(t_comb), t_target))
        return samples

    def __len__(self):
        # print("sum of data:", len(self.samples))
        return len(self.samples)

    def __getitem__(self, idx):
        time_indices, t_target = self.samples[idx]
        x = [self.data[t] for t in time_indices]
        y = self.data[t_target]
        score_last = self.score_list[t_target - 1]
        score = self.score_list[t_target]
        adj = self.adj_true_list[t_target]
        return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(y, dtype=torch.float32), score_last, score, adj


class ExhaustivePaddedGraphSequenceDataset(Dataset):
    def __init__(self, data_group_list, score_group_list, adj_group_list, num_node_list,
                 sequence_length=3, min_gap=1, target_offset=1, max_N=200):
        self.data_group_list = data_group_list
        self.score_group_list = score_group_list
        self.adj_group_list = adj_group_list
        self.num_node_list = num_node_list
        self.sequence_length = sequence_length
        self.min_gap = min_gap
        self.target_offset = target_offset
        self.max_N = max_N
        self.samples = self.build_all_samples()

    def build_all_samples(self):
        samples = []
        for group_id in range(len(self.data_group_list)):
            group_len = len(self.data_group_list[group_id])
            num_n = self.num_node_list[group_id]
            for t_comb in itertools.combinations(range(group_len), self.sequence_length):
                if all((t2 - t1 >= self.min_gap) for t1, t2 in zip(t_comb, t_comb[1:])):
                    t_target = t_comb[-1] + self.target_offset
                    if t_target < group_len:
                        samples.append((group_id, list(t_comb), t_target, num_n))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        group_id, time_indices, t_target, num_n = self.samples[idx]
        print("Index is:{}, {}, {}".format(group_id, time_indices, t_target))
        x_embed_seq = [self.data_group_list[group_id][t] for t in time_indices]
        y_embed = self.data_group_list[group_id][t_target]

        score_last = self.score_group_list[group_id][t_target - 1]
        score = self.score_group_list[group_id][t_target]

        adj_graph = self.adj_group_list[group_id][t_target]
        # adj_padded = graph_to_adj_tensor(adj_graph, self.max_N)
        adj_padded = self.pad_tensor(adj_graph, 2*self.max_N)

        # x_embed_seq = [self.pad_tensor(e, self.max_N) for e in x_embed_seq]
        # y_embed = self.pad_tensor(y_embed, self.max_N)

        x_tensor = torch.stack(x_embed_seq)
        return x_tensor, y_embed, score_last, score, adj_padded, num_n, group_id

    def pad_tensor(self, node_tensor, max_N):
        N, D = node_tensor.shape
        if N < max_N:
            padded = torch.zeros(max_N, max_N)
            padded[:N, :D] = node_tensor
            return padded
        return node_tensor


# 创建数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, score_list, adj_true_list, sequence_length=1):
        self.data = data
        self.score_list = score_list
        self.adj_true_list = adj_true_list
        self.seq_length = sequence_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        score_last = self.score_list[idx + self.seq_length - 1]
        score = self.score_list[idx + self.seq_length]
        adj = self.adj_true_list[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), score_last, score, adj


def build_graph_with_fixed_edges(A_pred_tensor, num_edges=200):
    """
    从 A_pred_tensor 构建一个具有指定数量边（无向图）的 NetworkX 图。
    """
    A_pred = A_pred_tensor.detach().cpu().numpy()
    N = A_pred.shape[0]

    # 提取上三角（不含对角）所有边的概率及其索引
    triu_indices = np.triu_indices(N, k=1)
    edge_probs = A_pred[triu_indices]

    # 选出最大概率的前 num_edges 条边
    topk_idx = np.argsort(edge_probs)[-num_edges:]
    selected_edges = [(triu_indices[0][i], triu_indices[1][i]) for i in topk_idx]

    # 构建空图并添加边
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(selected_edges)

    return G


def dtw_align_to_reference(ll, target_graphs):
    random.seed(42)
    indices = sorted(random.sample(range(1, len(target_graphs) - 1), ll - 2))
    indices = [0] + indices + [len(target_graphs) - 1]
    new_graphs = [target_graphs[i] for i in indices]
    return new_graphs


def dtw_align_to_reference_new(ll, target_graphs, target_scores):
    random.seed(42)
    indices = sorted(random.sample(range(1, len(target_graphs) - 1), ll - 2))
    indices = [0] + indices + [len(target_graphs) - 1]
    new_graphs = [target_graphs[i] for i in indices]
    new_scores = [target_scores[i] for i in indices]
    return new_graphs, new_scores


def pad_node_embeddings(node_embeds, max_N):
    N, D = node_embeds.shape
    padded = torch.zeros(max_N, D)
    padded[:N] = node_embeds
    return padded


def pad_networkx_graph(G, max_N=200):
    """
    将 G 补零到 max_N 个节点（以添加孤立节点的方式）
    保持返回仍为 NetworkX Graph
    """
    G_padded = G.copy()
    current_N = G.number_of_nodes()

    if current_N > max_N:
        raise ValueError(f"Graph has {current_N} nodes, which exceeds max_N={max_N}")

    for new_node in range(current_N, max_N):
        G_padded.add_node(new_node)  # 添加孤立节点
    for n in G_padded.nodes:
        G_padded.nodes[n]['valid'] = int(n < current_N)
        if n >= current_N:
            G_padded.nodes[n]['loc'] = [None, None]
            G_padded.nodes[n]['neighbor'] = []
    return G_padded


def unpad_networkx_graph(G_padded, original_N):
    """
    从补零后的 networkx 图中恢复出原始图（移除孤立 padding 节点）
    """
    nodes_to_keep = list(range(original_N))
    G_orig = G_padded.subgraph(nodes_to_keep).copy()
    return G_orig


class PrecomputedGraphDataset(Dataset):
    def __init__(self, path, device=torch.device("cpu")):
        raw_data = torch.load(path)
        self.device = device  # 新增参数，便于控制数据放在哪个设备上
        self.data_list = [
            (
                item["x_seq"].to(device),
                item["y"].to(device),
                torch.tensor(item["score_last"]).to(device),
                torch.tensor(item["score"]).to(device),
                item["adj"].to(device),
                torch.tensor(item["num_node"]).to(device),
                torch.tensor(item["group_id"]).to(device)
            )
            for item in raw_data
        ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def load_partial_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = model.state_dict()
    loaded_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
    model_state_dict.update(loaded_dict)
    model.load_state_dict(model_state_dict)
    print(f"\u2705 Loaded {len(loaded_dict)} / {len(model_state_dict)} layers from {checkpoint_path}")


def build_graph_tensor_topk(adj_matrix, num_edges):
    # adj_matrix: [N, N], assumed to be symmetric
    N = adj_matrix.size(0)
    mask = torch.triu(torch.ones_like(adj_matrix), diagonal=1)
    scores = adj_matrix[mask.bool()]
    topk_values, topk_indices = torch.topk(scores, num_edges)

    # map back to (i, j)
    row_idx, col_idx = torch.triu_indices(N, N, offset=1)
    selected_rows = row_idx[topk_indices]
    selected_cols = col_idx[topk_indices]

    # Create adjacency matrix
    edge_index = torch.stack([selected_rows, selected_cols], dim=0)
    adj_tensor = torch.zeros_like(adj_matrix)
    adj_tensor[selected_rows, selected_cols] = 1.0
    adj_tensor[selected_cols, selected_rows] = 1.0  # ensure symmetry
    return adj_tensor

