import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.autograd import Variable
import torch.cuda.amp as amp
import os
import time
import igraph as ig
import random
from random import randint
from collections import namedtuple
from contextlib import contextmanager
import psutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
import logging
import datetime
import time
import csv
import cProfile
import sys
import itertools  # Added for combinations in action space
import heapq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前设备: {device}")
if torch.cuda.is_available():
    print(f"CUDA可用: 使用 {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0) 
else:
    print("警告: 未检测到CUDA设备，将使用CPU运行")

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

LAPLACIAN_CACHE = {}  # 用于缓存拉普拉斯能量计算结果

logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    """
    上下文管理器，用于测量代码块的执行时间
    """
    try:
        start = time.perf_counter()
        yield
    finally:
        end = time.perf_counter()
        duration = end - start
        if name in phase_stats:
            phase_stats[name]["count"] += 1
            phase_stats[name]["total_time"] += duration

# 添加统计字典
phase_stats = {
    # Agent-C 统计
    "Agent-C Phase 1: Action Space Construction": {"count": 0, "total_time": 0},
    "Agent-C P1a: Neighbor Set Construction": {"count": 0, "total_time": 0},
    "Agent-C P1b: Candidate Pool Ranking": {"count": 0, "total_time": 0},
    "Agent-C P1c: Greedy Selection": {"count": 0, "total_time": 0},
    "Agent-C P1c1: S_v Computation": {"count": 0, "total_time": 0},
    "Agent-C P1c2a: Greedy FindMax": {"count": 0, "total_time": 0},
    "Agent-C P1c2b: Greedy ReduceSets": {"count": 0, "total_time": 0},
    "Agent-C Phase 1 (TopK-only)": {"count": 0, "total_time": 0},
    
    "Agent-C Phase 2: State Vector Construction (Pre-action)": {"count": 0, "total_time": 0},
    "Agent-C P2a: Delta Laplacian Energy": {"count": 0, "total_time": 0},
    "Agent-C P2b: Delta Edges": {"count": 0, "total_time": 0},
    "Agent-C P2c: Delta Overlap": {"count": 0, "total_time": 0},
    "Agent-C P2d: Clustering Coefficient": {"count": 0, "total_time": 0},

    "Agent-C Phase 3: Action Selection": {"count": 0, "total_time": 0},
    "Agent-C Phase 4: State Vector Construction (Post-action)": {"count": 0, "total_time": 0},
    "Agent-C Phase 5: Learning": {"count": 0, "total_time": 0},
    
    # Agent-R 统计
    "Agent-R Phase 1: Action Space Construction": {"count": 0, "total_time": 0},
    "Agent-R Phase 2: State Vector Construction (Pre-action)": {"count": 0, "total_time": 0},
    "Agent-R Phase 3: Action Selection": {"count": 0, "total_time": 0},
    "Agent-R Phase 4: State Vector Construction (Post-action)": {"count": 0, "total_time": 0},
    "Agent-R Phase 5: Learning": {"count": 0, "total_time": 0},
    "Agent-R Component Energy Calculation": {"count": 0, "total_time": 0},
}

@contextmanager
def phase_timer(name):
    """
    专门用于追踪各个阶段执行时间的计时器
    """
    try:
        start = time.perf_counter()
        yield
    finally:
        end = time.perf_counter()
        duration = end - start
        
        if name in phase_stats:
            phase_stats[name]["count"] += 1  # 累加计数
            phase_stats[name]["total_time"] += duration  # 累加时间

def calculate_laplacian_energy(subgraph, node_set, name_to_idx=None):
    """
    计算子图中包含的节点集合的拉普拉斯能量（igraph 版本，自动原编号映射）
    新定义：ℒ(G) = Σ(v∈G) Dv² + Σ(v∈G) Dv
    即：所有节点度数的平方和 + 所有节点的度数总和
    """
    with timer("Laplacian Energy Calculation"):
        if not node_set:  # 检查是否为空集合
            return 0

        # 创建缓存键
        node_tuple = tuple(sorted(node_set))
        cache_key = hash(node_tuple)

        # 检查缓存
        if cache_key in LAPLACIAN_CACHE:
            return LAPLACIAN_CACHE[cache_key]

        # === igraph部分自动原编号映射 ===
        # 始终使用当前子图的 name_to_idx，以避免索引失效
        current_name_to_idx = {v['name']: v.index for v in subgraph.vs}

        try:
            node_idx_list = [current_name_to_idx[x] for x in node_set if x in current_name_to_idx]
        except KeyError as e:
            raise ValueError(f"节点原编号不存在于当前子图中，请检查数据一致性。")

        # 创建指定节点的子图（用索引列表）
        temp_subgraph = subgraph.subgraph(node_idx_list)

        # 计算拉普拉斯能量：度数平方和 + 度数总和
        degree_sum = 0           # Σ Dv (度数总和)
        degree_square_sum = 0    # Σ Dv² (度数平方和)
        degrees = temp_subgraph.degree()
        for degree in degrees:
            degree_sum += degree
            degree_square_sum += degree * degree

        laplacian_energy = degree_square_sum + degree_sum

        # 存入缓存
        LAPLACIAN_CACHE[cache_key] = laplacian_energy

        # 限制缓存大小
        if len(LAPLACIAN_CACHE) > 10000:
            # 删除前5000个键
            keys = list(LAPLACIAN_CACHE.keys())[:5000]
            for k in keys:
                LAPLACIAN_CACHE.pop(k)

        return laplacian_energy

def evaluate_graph_preservation(original_graph, simplified_graph,name_to_idx=None):
    """
    评估简化图对原始图的保留程度（igraph版本，节点均用原编号）
    """
    # 节点集合（原编号）
    original_nodes = set(original_graph.vs["name"])
    simplified_nodes = set(simplified_graph.vs["name"])
    node_preservation = len(simplified_nodes) / len(original_nodes) if len(original_nodes) > 0 else 0

    # 边集合（原编号对，保证无向边唯一性）
    # def get_edge_set(graph):
    #     # 返回set，边为(sorted(src, tgt))元组
    #     return set(tuple(sorted((graph.vs[e.source]["name"], graph.vs[e.target]["name"]))) for e in graph.es)
    # original_edges = get_edge_set(original_graph)
    # simplified_edges = get_edge_set(simplified_graph)
    # edge_preservation = len(simplified_edges) / len(original_edges) if len(original_edges) > 0 else 0
    
    # 只计数，不关心具体边
    edge_preservation = simplified_graph.ecount() / original_graph.ecount() if original_graph.ecount() > 0 else 0
    # 拉普拉斯能量（直接用自动映射的calculate_laplacian_energy，传原编号集合）
    
    original_energy = calculate_laplacian_energy(original_graph, original_nodes,name_to_idx)
    simplified_energy = calculate_laplacian_energy(simplified_graph, simplified_nodes,name_to_idx)
    laplacian_energy_preservation = simplified_energy / original_energy if original_energy > 0 else 0

    return {
        'node_preservation': node_preservation,
        'edge_preservation': edge_preservation,
        'laplacian_energy_preservation': laplacian_energy_preservation
    }

def delta_laplacian_energy(v, candidate_set, subgraph, cached_energy=None,name_to_idx=None):
    """
    计算节点v从候选集合中删除/添加后的拉普拉斯能量变化（igraph+原编号适配版）
    ℒ(G) = Σ(v∈G) Dv² + Σ(v∈G) Dv

    参数:
    v: 要评估的节点（原编号）
    candidate_set: 当前已收集的候选节点集合（原编号集合）
    subgraph: 子图（要求vs["name"]为原编号）
    cached_energy: 缓存的原始能量值（float），可选

    返回:
    (能量增量, 新集合能量),由于set和v都是原始编号且在计算中映射，因此不需要映射
    """
    # 始终使用当前子图的 name_to_idx
    current_name_to_idx = {v['name']: v.index for v in subgraph.vs}

    if cached_energy is None:
        # 计算当前候选集合的能量（原编号，自动映射）
        original_energy_C = calculate_laplacian_energy(subgraph, candidate_set, current_name_to_idx)
    else:
        original_energy_C = cached_energy

    # 新集合：添加节点v
    new_node_set = set(candidate_set) | {v}
    new_energy_C = calculate_laplacian_energy(subgraph, new_node_set, current_name_to_idx)

    return new_energy_C - original_energy_C, new_energy_C

def delta_edges(v, candidate_set, subgraph, name_to_idx=None):
    """
    计算在已收集节点子图中，添加一个新节点 v 后新增的边数（igraph+原编号）
    """
    # 始终使用当前子图的 name_to_idx
    current_name_to_idx = {v['name']: v.index for v in subgraph.vs}

    # 使用当前映射获取 idx_v
    idx_v = current_name_to_idx.get(v)
    if idx_v is None:
        raise ValueError(f"节点 {v} 不存在于当前子图中")

    idx_candidate = [current_name_to_idx.get(x) for x in candidate_set if x in current_name_to_idx]

    # 当前已收集节点的子图
    temp_subgraph = subgraph.subgraph(idx_candidate)
    existing_edges = temp_subgraph.ecount()

    # 模拟加入v后的边
    neighbors_of_v = set(subgraph.neighbors(idx_v))
    candidate_indices = set(idx_candidate)
    edges_to_add = [(idx_v, n) for n in neighbors_of_v if n in candidate_indices]

    total_edges = existing_edges + len(edges_to_add)
    return total_edges - existing_edges

def delta_overlap(v, neighbors_set, subgraph, name_to_idx=None):
    """
    计算节点 v 与其他候选节点的连接数（igraph+原编号）
    Parameters:
        v: 待评估的节点（原编号）
        neighbors_set: 已收集节点的所有邻居节点集合（原编号集合）
        subgraph: 图对象（vs["name"]为原编号）
    """
    # 始终使用当前子图的 name_to_idx
    current_name_to_idx = {v['name']: v.index for v in subgraph.vs}

    # 使用当前映射获取 idx_v
    idx_v = current_name_to_idx.get(v)
    if idx_v is None:
        raise ValueError(f"节点 {v} 不存在于当前子图中")

    idx_neighbors_set = set(current_name_to_idx.get(x) for x in neighbors_set if x in current_name_to_idx)
    connections = [n for n in subgraph.neighbors(idx_v) if n in idx_neighbors_set and n != idx_v]
    return len(connections)

def calculate_clustering_coefficient(v, candidate_set, subgraph, name_to_idx=None):
    """
    计算节点v的聚类系数（igraph+原编号，完全自动映射）
    """
    with phase_timer("Agent-C P2d: Clustering Coefficient"):
        # 始终使用当前子图的 name_to_idx
        current_name_to_idx = {v['name']: v.index for v in subgraph.vs}

        # 使用当前映射获取 idx_v
        idx_v = current_name_to_idx.get(v)
        if idx_v is None:
            raise ValueError(f"节点 {v} 不存在于当前子图中")

        # 使用igraph的local transitivity
        cc = subgraph.transitivity_local_undirected(vertices=[idx_v])
        # transitivity_local_undirected 返回的是一个列表，取第一个
        return cc[0] if cc else 0.0

def build_state_vector_C(subgraph, action_space, candidate_set, action_space_size=None,name_to_idx=None,
                         degrees_cache=None, neighbors_idx_cache=None, clustering_cache=None, verbose=False):
    """
    Agent_C的状态向量构建函数 - 收集阶段（igraph兼容、内部索引化、高性能版）

    参数：
        subgraph: igraph.Graph对象，vs["name"]为原编号
        action_space: list，动作空间（每个元素为节点原编号，或原编号列表/元组）
        candidate_set: set，已收集节点集合（原编号）
        action_space_size: 固定动作空间节点数
    返回：
        torch.FloatTensor, shape=(k1_fixed * 4,)
    """

    with phase_timer("Agent-C Phase 2: State Vector Construction (Pre-action)"):
        k1_fixed = action_space_size or 5  # 默认值为5
        
        # 获取所有候选节点（原编号），去重、排序
        all_candidate_nodes = set()
        for action in action_space:
            if isinstance(action, (list, tuple)):
                all_candidate_nodes.update(action)
            elif action is not None:
                all_candidate_nodes.add(action)
        candidate_nodes_list = sorted(list(all_candidate_nodes))
        k1_actual = len(candidate_nodes_list)
        
        # print(f"构建状态向量：实际候选节点数={k1_actual}, 固定维度k1={k1_fixed}")
        # print(f"候选节点: {candidate_nodes_list}")

        state_dim = k1_fixed * 4
        state_vector = []

        # 始终使用当前子图的 name_to_idx（允许外部缓存）
        current_name_to_idx = {v['name']: v.index for v in subgraph.vs}

        # candidate_set 索引版
        candidate_set_idx = set(current_name_to_idx.get(x) for x in candidate_set if x in current_name_to_idx)
        # 预先计算当前候选集合的拉普拉斯能量（如有）——若使用快速路径，将不再用到
        current_energy = calculate_laplacian_energy(subgraph, candidate_set, current_name_to_idx) if (candidate_set and (degrees_cache is None or neighbors_idx_cache is None)) else 0

        # 若提供缓存，准备索引化集合与增量数据
        use_fast_path = degrees_cache is not None and neighbors_idx_cache is not None
        if use_fast_path:
            candidate_set_idx = set(current_name_to_idx.get(x) for x in candidate_set if x in current_name_to_idx)
            # 在当前 S 上一次性计算 deg_in_S，用于 ΔL 快速计算
            deg_in_S = {u: len(neighbors_idx_cache[u] & candidate_set_idx) for u in candidate_set_idx}

        for i in range(k1_fixed):
            if i < len(candidate_nodes_list):
                node = candidate_nodes_list[i]

                # 使用当前映射获取 idx_node
                idx_node = current_name_to_idx.get(node)
                if idx_node is None:
                    raise ValueError(f"节点 {node} 不存在于当前子图中")

                # 特征1~4：使用缓存的快速路径或回退到原有实现
                if use_fast_path:
                    # 计算索引集合
                    S_idx = candidate_set_idx
                    Nv = neighbors_idx_cache[idx_node]
                    Ns = Nv & S_idx
                    k = len(Ns)
                    # ΔL 快速公式： (k + k^2) + Σ_{u∈Ns} (2*deg_in_S[u] + 2)
                    with phase_timer("Agent-C P2a: Delta Laplacian Energy"):
                        delta_energy = (k + k * k) + sum(2 * deg_in_S.get(u, 0) + 2 for u in Ns)

                    # 新增边数：Nv 与 S 的交集大小
                    with phase_timer("Agent-C P2b: Delta Edges"):
                        delta_edge = len(Ns)

                    # 重叠：deg(v) - 边指向 S 的条数（与原定义等价）
                    with phase_timer("Agent-C P2c: Delta Overlap"):
                        deg_v = degrees_cache[idx_node] if degrees_cache is not None else subgraph.degree(idx_node)
                        overlap = float(deg_v - delta_edge)

                    # 聚类系数：直接查缓存
                    with phase_timer("Agent-C P2d: Clustering Coefficient"):
                        if clustering_cache is not None and 0 <= idx_node < len(clustering_cache):
                            clustering = float(clustering_cache[idx_node]) if clustering_cache[idx_node] is not None else 0.0
                        else:
                            clustering = calculate_clustering_coefficient(node, candidate_set, subgraph, current_name_to_idx)
                            if isinstance(clustering, tuple):
                                clustering = float(clustering[0])
                else:
                    # 回退到原有实现
                    with phase_timer("Agent-C P2a: Delta Laplacian Energy"):
                        delta_energy = delta_laplacian_energy(node, candidate_set, subgraph, current_energy, current_name_to_idx)
                    if isinstance(delta_energy, tuple):
                        delta_energy = float(delta_energy[0])

                    with phase_timer("Agent-C P2b: Delta Edges"):
                        delta_edge = delta_edges(node, candidate_set, subgraph, current_name_to_idx)
                    if isinstance(delta_edge, tuple):
                        delta_edge = int(delta_edge[0])

                    node_neighbors_idx = set(subgraph.neighbors(idx_node))
                    node_neighbors = set(subgraph.vs[n]["name"] for n in node_neighbors_idx if n not in candidate_set_idx)
                    with phase_timer("Agent-C P2c: Delta Overlap"):
                        overlap = delta_overlap(node, node_neighbors, subgraph, current_name_to_idx)
                    if isinstance(overlap, tuple):
                        overlap = float(overlap[0])

                    with phase_timer("Agent-C P2d: Clustering Coefficient"):
                        clustering = calculate_clustering_coefficient(node, candidate_set, subgraph, current_name_to_idx)
                    if isinstance(clustering, tuple):
                        clustering = float(clustering[0])

                if verbose:
                    print(f"节点{node}的特征值:")
                    print(f"- Δℒ={float(delta_energy):.3f}")
                    print(f"- Δedges={int(delta_edge)}")
                    print(f"- overlap={float(overlap):.3f}")
                    print(f"- clustering={float(clustering):.3f}")

                state_vector.extend([
                    float(delta_energy),
                    float(delta_edge),
                    float(overlap),
                    float(clustering)
                ])
            else:
                # 填充零
                state_vector.extend([0.0, 0.0, 0.0, 0.0])
                if i == len(candidate_nodes_list):
                    print(f"用零填充 {k1_fixed - k1_actual} 个节点位置")

        assert len(state_vector) == state_dim, f"状态向量长度 {len(state_vector)} 不等于期望长度 {state_dim}"

        print(f"Agent-C状态向量维度: {len(state_vector)} (固定k1={k1_fixed}, 每个节点4维特征)")

        state_vector = [float(x) for x in state_vector]
        return torch.FloatTensor(state_vector).to(device)

def build_collection_action_space_new(current_neighbors_set, subgraph, name_to_idx, action_space_size=5, S_v=5, degrees_cache=None, greedy=True, collection_mode="greedy", window_set=None, window_size=None):
    """
    Builds the action space for Agent-C using the S_v mechanism.
    (INCREMENTAL VERSION)
    """
    with phase_timer("Agent-C Phase 1: Action Space Construction"):
        # Step 1: Use the pre-computed neighbors set directly
        # The expensive P1a (Neighbor Set Construction) is now handled outside in the main loop.
        if not current_neighbors_set:
            return []

        current_name_to_idx = {v['name']: v.index for v in subgraph.vs}

        # Step 2: Build candidate_pool according to collection_mode
        with phase_timer("Agent-C P1b: Candidate Pool Ranking"):
            # Effective window set for greedy_window mode
            if collection_mode == "greedy_window":
                effective_window = set(window_set) if window_set is not None else set(current_neighbors_set)
                # 限制到当前邻居集合
                effective_window &= set(current_neighbors_set)
            else:
                effective_window = set(current_neighbors_set)

            if degrees_cache is not None:
                # 选择 Top-k：topk 模式取 k1；greedy 模式取 S_v；greedy_window 使用窗口内 Top
                if collection_mode == "topk":
                    top_k = action_space_size
                else:
                    top_k = S_v
                candidate_pool = heapq.nlargest(
                    top_k,
                    list(effective_window),
                    key=lambda n: degrees_cache[current_name_to_idx.get(n, -1)] if current_name_to_idx.get(n, -1) != -1 else -1
                )
            else:
                sort_base = list(effective_window)
                if collection_mode == "topk":
                    top_k = action_space_size
                else:
                    top_k = S_v
                candidate_pool = sorted(
                    sort_base,
                    key=lambda n: subgraph.degree(current_name_to_idx.get(n, -1)),
                    reverse=True
                )[:top_k]

        # TopK-only 分支：跳过贪心，直接用 Top-k1 的前缀
        # 提前定义 k1，供后续动作空间生成使用
        k1 = action_space_size

        if collection_mode == "topk":
            with phase_timer("Agent-C Phase 1 (TopK-only)"):
                # 直接使用度数最高的 k1 个节点
                prioritized_nodes = candidate_pool[:action_space_size]
        elif collection_mode in ("greedy", "greedy_window"):
            # Adapted greedy selection from highRL: Compute S_v for each in candidate_pool
            with phase_timer("Agent-C P1c: Greedy Selection"):
                with phase_timer("Agent-C P1c1: S_v Computation"):
                    remaining_S_v = {}
                    for v in candidate_pool:
                        idx_v = current_name_to_idx.get(v, -1)
                        if idx_v == -1:
                            continue
                        neighbors_idx = subgraph.neighbors(idx_v)
                        S_v_set = set(subgraph.vs[n]['name'] for n in neighbors_idx)
                        if collection_mode == "greedy_window":
                            # 仅在窗口内统计 S_v
                            window_for_intersection = set(window_set) if window_set is not None else set(current_neighbors_set)
                            S_v_set = S_v_set & window_for_intersection
                        remaining_S_v[v] = S_v_set.copy()

                # Greedy selection loop (multi-round max |S_v|, fallback to degree if all <=0)
                with phase_timer("Agent-C P1c2a: Greedy FindMax"):
                    # 此计时仅包找最大；差集放到 ReduceSets
                    pass
                with phase_timer("Agent-C P1c2b: Greedy ReduceSets"):
                    pass

                # 具体贪心逻辑，拆分计时：
                prioritized_nodes = []
                remaining_nodes = set(candidate_pool)
                for round_num in range(k1):
                    if not remaining_nodes:
                        break

                    # Find max |S_v| node
                    with phase_timer("Agent-C P1c2a: Greedy FindMax"):
                        max_sv_size = max([len(remaining_S_v.get(node, set())) for node in remaining_nodes], default=0)
                        if max_sv_size > 0:
                            valid_candidates = [(node, len(remaining_S_v[node])) for node in remaining_nodes if node in remaining_S_v]
                            max_node = max(valid_candidates, key=lambda x: x[1])[0]
                            max_set = remaining_S_v[max_node].copy()
                        else:
                            max_node = None
                            max_set = set()

                    if max_node is not None:
                        prioritized_nodes.append(max_node)
                        del remaining_S_v[max_node]
                        with phase_timer("Agent-C P1c2b: Greedy ReduceSets"):
                            for node in list(remaining_S_v.keys()):
                                remaining_S_v[node] -= max_set
                        remaining_nodes.remove(max_node)
                    else:
                        # Fallback to degree sorting
                        if remaining_nodes:
                            next_node = max(remaining_nodes, key=lambda x: subgraph.degree(current_name_to_idx.get(x, -1)))
                            prioritized_nodes.append(next_node)
                            remaining_nodes.remove(next_node)

        # Generate action space as cumulative prefixes (adapted from highRL)
        with phase_timer("Agent-C: Action Space Generation"):
            action_space = []
            for j in range(1, k1 + 1):
                selected_count = min(j, len(prioritized_nodes))
                if selected_count == 0:
                    action_space.append([])
                else:
                    action_space.append(prioritized_nodes[:selected_count])

        return action_space

def build_selection_action_space_multi(candidate_nodes, core_graph_nodes, subgraph, action_space_size, k2=None, name_to_idx=None,
                                       degrees_cache=None, neighbors_idx_cache=None):
    """
    为Agent-R构建选择动作空间（多节点版本，igraph兼容，节点全为原编号）。
    """
    with phase_timer("Agent-R Phase 1: Action Space Construction"):
        actual_k2 = k2 if k2 is not None else action_space_size

        if not candidate_nodes:
            return [[] for _ in range(actual_k2)], [0.0] * actual_k2, [0] * actual_k2, set()

        print(f"\n=== Agent-R动作空间构建过程 ===")
        print(f"Agent-C传递的候选节点: {candidate_nodes}")
        print(f"动作空间大小参数: action_space_size={action_space_size}, k2={actual_k2}")

        # 始终使用当前子图的 name_to_idx
        current_name_to_idx = {v['name']: v.index for v in subgraph.vs}

        # 按度数排序候选节点（度数高的在前，原编号）
        if degrees_cache is not None:
            candidate_nodes_sorted = sorted(candidate_nodes,
                                            key=lambda x: degrees_cache[current_name_to_idx.get(x)],
                                            reverse=True)
        else:
            candidate_nodes_sorted = sorted(candidate_nodes,
                                           key=lambda x: subgraph.degree(current_name_to_idx.get(x)),
                                           reverse=True)

        # 检查是否有无效节点
        for x in candidate_nodes:
            if current_name_to_idx.get(x) is None:
                raise ValueError(f"节点 {x} 不存在于当前子图中")

        if degrees_cache is not None:
            candidate_degrees = [(node, degrees_cache[current_name_to_idx.get(node)]) for node in candidate_nodes_sorted]
        else:
            candidate_degrees = [(node, subgraph.degree(current_name_to_idx.get(node))) for node in candidate_nodes_sorted]
        print(f"按度数排序后的节点序列: {candidate_degrees}")
        print(f"即：v1={candidate_nodes_sorted[0]}(度数最高), ..., vj={candidate_nodes_sorted[-1]}(度数最低)")

        # 使用快速路径时不需要整体能量
        current_energy = calculate_laplacian_energy(subgraph, core_graph_nodes, current_name_to_idx) if neighbors_idx_cache is None else 0
        
        j = len(candidate_nodes_sorted)

        all_prefix_sets = []
        all_deltas = []
        all_descriptions = []

        print(f"\n开始生成所有前缀集合并计算能量增益:")

        # 预处理核心索引集合与核心内度（用于快速增量）
        if neighbors_idx_cache is not None:
            C_idx = set(current_name_to_idx.get(x) for x in core_graph_nodes if x in current_name_to_idx)
            # 初始化核心内度
            deg_in_core = {}
            for u in C_idx:
                deg_in_core[u] = len(neighbors_idx_cache[u] & C_idx)

        for start_idx in range(j):
            if neighbors_idx_cache is None:
                # 回退路径：按原方法逐集合重算
                for length in range(1, j - start_idx + 1):
                    node_set = set(candidate_nodes_sorted[start_idx:start_idx + length])

                    start_node_idx = start_idx + 1  # 1-based
                    if start_idx == 0:
                        set_name = f"S1{length}"
                        description = f"S1{length} = {{v1, v2, ..., v{length}}}"
                    elif length == 1:
                        set_name = f"S{start_node_idx}1"
                        description = f"S{start_node_idx}1 = {{v{start_node_idx}}}"
                    else:
                        set_name = f"S{start_node_idx}{length}"
                        end_node_idx = start_idx + length
                        description = f"S{start_node_idx}{length} = {{v{start_node_idx}, ..., v{end_node_idx}}}"

                    print(f"\n--- 计算前缀集合 {set_name} ---")
                    print(f"  {description}")
                    print(f"  节点集合: {sorted(node_set)}")

                    new_core_nodes = core_graph_nodes | node_set
                    new_energy = calculate_laplacian_energy(subgraph, new_core_nodes, current_name_to_idx)
                    energy_gain = new_energy - current_energy

                    print(f"  能量增益 Δ{set_name} = {energy_gain:.6f}")

                    all_prefix_sets.append(node_set)
                    all_deltas.append(energy_gain)
                    all_descriptions.append((set_name, description))
            else:
                # 快速路径：对固定 start 增量式累加
                start_node_idx = start_idx + 1  # 1-based
                # 在线维护度：从核心内度复制
                deg_temp = dict(deg_in_core)
                added = set()
                current_gain = 0.0

                for length in range(1, j - start_idx + 1):
                    # 取本次要加入的节点（原编号 -> 索引）
                    v_name = candidate_nodes_sorted[start_idx + length - 1]
                    idx_v = current_name_to_idx.get(v_name)
                    # 边界集合：v 与 (核心∪已加入) 的邻接
                    Ns = neighbors_idx_cache[idx_v] & (C_idx | added)
                    k = len(Ns)
                    # 增量 Δ(v|X)
                    delta_v = (k + k * k) + sum(2 * deg_temp.get(u, 0) + 2 for u in Ns)
                    current_gain += delta_v
                    # 更新度
                    for u in Ns:
                        deg_temp[u] = deg_temp.get(u, 0) + 1
                    deg_temp[idx_v] = k
                    added.add(idx_v)

                    # 记录集合与增益
                    node_set = set(candidate_nodes_sorted[start_idx:start_idx + length])
                    if start_idx == 0:
                        set_name = f"S1{length}"
                        description = f"S1{length} = {{v1, v2, ..., v{length}}}"
                    elif length == 1:
                        set_name = f"S{start_node_idx}1"
                        description = f"S{start_node_idx}1 = {{v{start_node_idx}}}"
                    else:
                        set_name = f"S{start_node_idx}{length}"
                        end_node_idx = start_idx + length
                        description = f"S{start_node_idx}{length} = {{v{start_node_idx}, ..., v{end_node_idx}}}"

                    print(f"\n--- 计算前缀集合 {set_name} (增量) ---")
                    print(f"  {description}")
                    print(f"  节点集合: {sorted(node_set)}")
                    print(f"  能量增益 Δ{set_name} = {current_gain:.6f}")

                    all_prefix_sets.append(node_set)
                    all_deltas.append(current_gain)
                    all_descriptions.append((set_name, description))

        print(f"\n=== 第5步：对所有能量增益进行排序 ===")
        print(f"总共生成了 {len(all_prefix_sets)} 个前缀集合")

        sorted_indices = sorted(range(len(all_deltas)), key=lambda i: all_deltas[i], reverse=True)

        print(f"\n所有前缀集合的能量增益排序（从高到低）:")
        for rank, idx in enumerate(sorted_indices, 1):
            set_name, description = all_descriptions[idx]
            energy_gain = all_deltas[idx]
            nodes = sorted(all_prefix_sets[idx])
            print(f"  第{rank}名: {set_name}, Δ={energy_gain:.6f}, 节点={nodes}")

        available_actions = len(all_prefix_sets)
        print(f"\n=== 选择前{available_actions}个最优动作并填充到k2={actual_k2}个动作 ===")

        action_space = []
        valid_deltas = []
        valid_sizes = []
        selected_descriptions = []

        for i, idx in enumerate(sorted_indices):
            action_space.append(list(all_prefix_sets[idx]))
            energy_gain = all_deltas[idx]
            valid_deltas.append(energy_gain)
            valid_sizes.append(len(all_prefix_sets[idx]))
            selected_descriptions.append(all_descriptions[idx])
            set_name, description = all_descriptions[idx]
            print(f"  动作{i+1}: {set_name}, Δ={energy_gain:.6f}, {description}")

        if len(action_space) < actual_k2:
            padding_needed = actual_k2 - len(action_space)
            print(f"\n动作数量不足k2，需要填充{padding_needed}个动作")
            if len(action_space) > 0:
                first_action = action_space[0]
                first_delta = valid_deltas[0]
                first_size = valid_sizes[0]
                first_description = selected_descriptions[0]
                for i in range(padding_needed):
                    action_index = len(action_space) + 1
                    action_space.append(first_action.copy())
                    valid_deltas.append(first_delta)
                    valid_sizes.append(first_size)
                    selected_descriptions.append((f"复制{first_description[0]}", f"复制的{first_description[1]}"))
                    print(f"  动作{action_index}: 复制{first_description[0]}, Δ={first_delta:.6f}, 节点={sorted(first_action)}")
            else:
                for i in range(actual_k2):
                    action_space.append([])
                    valid_deltas.append(0.0)
                    valid_sizes.append(0)
                    selected_descriptions.append(("空动作", "空动作"))
                    print(f"  动作{i+1}: 空动作, Δ=0.0, 节点=[]")
        elif len(action_space) > actual_k2:
            print(f"\n动作数量超过k2，只保留前{actual_k2}个动作")
            action_space = action_space[:actual_k2]
            valid_deltas = valid_deltas[:actual_k2]
            valid_sizes = valid_sizes[:actual_k2]
            selected_descriptions = selected_descriptions[:actual_k2]

        accepted_nodes = set()
        for action_set in action_space:
            accepted_nodes.update(action_set)

        all_candidate_nodes = set(candidate_nodes_sorted)
        rejected_nodes = all_candidate_nodes - accepted_nodes

        print(f"\n=== 动作空间构建完成 ===")
        print(f"最终动作空间大小: {len(action_space)} (确保k2={actual_k2})")
        print(f"被接受的节点: {sorted(accepted_nodes) if accepted_nodes else '无'}")
        print(f"被拒绝的节点: {sorted(rejected_nodes) if rejected_nodes else '无'}")
        print(f"=== Agent-R动作空间构建结束 ===\n")

        assert len(action_space) == actual_k2, f"动作空间大小 {len(action_space)} 不等于 k2 {actual_k2}"
        assert len(valid_sizes) == actual_k2, f"集合大小列表大小 {len(valid_sizes)} 不等于 k2 {actual_k2}"

        return action_space, valid_deltas, valid_sizes, rejected_nodes

def build_state_vector_R_multi(subgraph, action_space, valid_deltas, valid_sizes, core_graph_nodes, action_space_size=5, k2=None):
    """
    为Agent-R构建状态向量（多节点版本，igraph兼容，无需索引映射）。
    返回格式 [Δ1, |S1|, Δ2, |S2|, ..., Δk2, |Sk2|]
    """
    import torch
    with phase_timer("Agent-R Phase 2: State Vector Construction (Pre-action)"):
        actual_k2 = k2 if k2 is not None else action_space_size

        # 若动作空间为空，直接返回全零向量
        if not action_space or not valid_deltas or not valid_sizes:
            return torch.zeros(actual_k2 * 2, dtype=torch.float32)

        # 断言一致性
        assert len(action_space) == actual_k2, f"动作空间大小 {len(action_space)} 不等于 k2 {actual_k2}"
        assert len(valid_deltas) == actual_k2, f"能量增益列表大小 {len(valid_deltas)} 不等于 k2 {actual_k2}"
        assert len(valid_sizes) == actual_k2, f"集合大小列表大小 {len(valid_sizes)} 不等于 k2 {actual_k2}"

        # logger.info(f"\n=== 第7步：构建Agent-R状态向量 ===")
        # logger.info(f"状态格式: (c(S1), c(S2), ..., c(Sk2))，其中k2={actual_k2}")
        # logger.info(f"其中c(Si) = (Δi, |Si|)，即添加集合Si后的总能量增益和增加的节点个数")

        state_vector_list = []
        for i, (action_set, delta_i, size_i) in enumerate(zip(action_space, valid_deltas, valid_sizes)):
            state_vector_list.append(delta_i)
            state_vector_list.append(float(size_i))
            # logger.debug(f"动作{i+1}: 节点集合 = {sorted(action_set)}")
            # logger.debug(f"        c(S{i+1}) = (Δ{i+1}, |S{i+1}|) = ({delta_i:.6f}, {size_i})")

        state_vector = torch.tensor(state_vector_list, dtype=torch.float32)
        expected_dim = actual_k2 * 2

        # logger.info(f"\nAgent-R状态向量: {state_vector.tolist()}")
        # logger.info(f"状态向量维度: {state_vector.shape}")
        # logger.info(f"期望维度: {expected_dim} (k2 × 2 = {actual_k2} × 2)")
        # logger.info(f"格式: [Δ1, |S1|, Δ2, |S2|, ..., Δk2, |Sk2|] (每对值对应一个集合动作的二元组)")
        # logger.info(f"注意: Δi为总能量增益，符合理论要求")
        # logger.info(f"=== Agent-R状态向量构建完成 ===\n")

        assert state_vector.shape[0] == expected_dim, f"状态向量维度 {state_vector.shape[0]} 与期望维度 {expected_dim} 不匹配"
        return state_vector


# one_layer_nn and Agent are for insertion
class one_layer_nn(nn.Module):
    def __init__(self, ALPHA, input_size, hidden_size1, output_size, is_gpu=True):
        super(one_layer_nn, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1, bias=False)
        self.expected_input_dim = input_size  # 记录期望输入维度
        self.layer2 = nn.Linear(hidden_size1, output_size, bias=False)
        self.activation = nn.SELU()  # 新增激活函数定义

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA) # learning rate
        self.loss = nn.MSELoss()
        # 使用全局device而不是条件判断
        self.device = device
        self.to(self.device)
        
        # 添加混合精度训练支持
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        print(f"Model moved to: {self.device}")

    def forward(self, state):
        # sm = nn.Softmax(dim=0)
        sm = nn.SELU() # SELU activation function
        #当前激活函数是SELU，需要修改吗

        # Variable is used for older torch
        x = Variable(state, requires_grad=False).to(self.device)
        
        # 检查输入维度是否正确
        expected_input_size = self.layer1.weight.size(1)
        
        # 如果是一维张量，添加batch维度
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # 检查特征维度是否匹配
        if x.size(1) != expected_input_size:
            print(f"警告: 输入特征维度 {x.size(1)} 与模型期望的 {expected_input_size} 不匹配")
            # 调整维度
            if x.size(1) < expected_input_size:
                # 不足则补零
                padding = torch.zeros(x.size(0), expected_input_size - x.size(1), device=self.device)
                x = torch.cat([x, padding], dim=1)
            else:
                # 过多则截断
                x = x[:, :expected_input_size]
                
        y = self.layer1(x)
        y_hat = sm(y)
        z = self.layer2(y_hat)
        scores = z
        return scores


class BaseAgent():
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, batch_size, action_space_size, dim, is_gpu, epsEnd=0.1, replace=20):
        # epsEnd was set to be 0.05 initially
        self.GAMMA = gamma  # discount factor
        self.EPSILON = epsilon  # epsilon greedy to choose action
        self.EPS_END = epsEnd  # minimum epsilon
        self.memSize = maxMemorySize  # memory size
        self.batch_size = batch_size  # 训练时使用的批量大小
        self.steps = 0  # 总步数
        self.learn_step_counter = 0  # 学习步骤计数器
        self.memory = []  # 列表存储经验回放缓冲区
        self.memCntr = 0  # 经验回放缓冲区计数器
        self.replace_target_cnt = replace  # 目标网络更新频率
        self.action_space_size = action_space_size  # 动作空间大小
        self.Q_eval = one_layer_nn(
            ALPHA=alpha, 
            input_size=dim, 
            hidden_size1=32, 
            output_size=action_space_size,
            is_gpu=is_gpu
        )
        self.Q_target = one_layer_nn(
            ALPHA=alpha, 
            input_size=dim, 
            hidden_size1=32, 
            output_size=action_space_size,
            is_gpu=is_gpu
        )
        # 添加性能监控计数器
        self.forward_time = 0    # 记录前向传播总时间
        self.learn_time = 0      # 记录学习总时间
        self.action_count = 0    # 记录动作选择次数
        self.learn_count = 0     # 记录学习次数
        
        # 添加损失跟踪
        self.total_loss = 0.0
        self.batch_count = 0
        self.epoch_losses = []
        self.training = True  # 添加训练模式标志
    
    def storeTransition(self, state, action, reward, state_):  # 存储经验到经验回放缓冲区
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr % self.memSize] = [state, action, reward, state_]
        self.memCntr = self.memCntr + 1

    def sample(self):  # 从经验回放缓冲区中随机采样一批经验
        if len(self.memory) < self.batch_size:
            return self.memory
        else:
            return random.sample(self.memory, self.batch_size)

    def eval(self):
        """设置为推理模式"""
        self.training = False
        self.EPSILON = 0.0  # 在推理模式下禁用探索
        self.Q_eval.eval()  # 设置神经网络为评估模式
        if hasattr(self, 'Q_target'):
            self.Q_target.eval()
    
    def train(self):
        """设置为训练模式"""
        self.training = True
        self.Q_eval.train()  # 设置神经网络为训练模式
        if hasattr(self, 'Q_target'):
            self.Q_target.train()
    
    def chooseAction(self, observation):
        """修改动作选择逻辑以支持推理模式"""
        with phase_timer(f"{self.__class__.__name__} Phase 3: Action Selection"):
            if not isinstance(observation, torch.Tensor):
                state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
            else:
                state = observation.to(self.Q_eval.device)
            
            if np.random.random() > self.EPSILON or not self.training:
                # 在推理模式或不进行探索时，直接使用最优动作
                with torch.no_grad():
                    actions = self.Q_eval.forward(state)
                    action = torch.argmax(actions).item()
            else:
                # 在训练模式且进行探索时，随机选择动作
                action = np.random.randint(0, self.action_space_size)
            
            return action

    def learn(self):
        """
        从经验回放缓冲区中学习
        """
        with phase_timer(f"{self.__class__.__name__} Phase 5: Learning"):
            if len(self.memory) <= 0:
                return
            
            self.Q_eval.optimizer.zero_grad()
            if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
                self.Q_target.load_state_dict(self.Q_eval.state_dict())
            
            # 从内存中获取数据样本
            transitions = self.sample()
            batch = list(zip(*transitions))  # 将列表转置

            state_tensor = torch.stack(list(batch[0]))
            actions = torch.tensor(batch[1], dtype=torch.int64).to(device)
            rewards = torch.tensor(batch[2], dtype=torch.float32).to(device)
            next_state_tensor = torch.stack(list(batch[3]))

            # 确保所有张量都在同一个设备上，以防万一
            if state_tensor.device != device:
                state_tensor = state_tensor.to(device)
            if next_state_tensor.device != device:
                next_state_tensor = next_state_tensor.to(device)
           
            # 使用混合精度训练 - 修改判断条件，确保CUDA可用
            if torch.cuda.is_available() and self.Q_eval.scaler is not None:
                with amp.autocast():
                    state_action_values = self.Q_eval.forward(state_tensor)
                    next_state_action_values = self.Q_target.forward(next_state_tensor)
             
                    # 批量计算目标值
                    max_value, _ = torch.max(next_state_action_values, 1)
                    target_values = rewards + self.GAMMA * max_value
                    
                    # 创建目标Q值
                    state_action_values_target = state_action_values.clone()
                    
                    # 使用索引更新，避免循环
                    batch_indices = torch.arange(len(actions), device=device)
                    # 确保目标值类型与模型输出类型匹配
                    target_values = target_values.to(state_action_values.dtype)
                    state_action_values_target[batch_indices, actions] = target_values
    
                    loss = self.Q_eval.loss(state_action_values, state_action_values_target.detach())
                    
                    # 记录损失值
                    loss_value = loss.item()
                    self.total_loss += loss_value
                    self.batch_count += 1
                    
                    logger.info(f"Batch Loss [{self.__class__.__name__}]: {loss_value:.6f}, Average Loss: {self.get_average_loss():.6f}")
                
                # 使用scaler进行反向传播和优化
                self.Q_eval.scaler.scale(loss).backward()
                self.Q_eval.scaler.step(self.Q_eval.optimizer)
                self.Q_eval.scaler.update()
            else:
                # 原始的非混合精度训练，也优化了实现
                state_action_values = self.Q_eval.forward(state_tensor)
                next_state_action_values = self.Q_target.forward(next_state_tensor)
         
                # 批量计算目标值
                max_value, _ = torch.max(next_state_action_values, 1)
                target_values = rewards + self.GAMMA * max_value
                
                # 创建目标Q值
                state_action_values_target = state_action_values.clone()
                
                # 使用索引更新，避免循环
                batch_indices = torch.arange(len(actions), device=device)
                # 确保目标值类型与模型输出类型匹配
                target_values = target_values.to(state_action_values.dtype)
                state_action_values_target[batch_indices, actions] = target_values

                loss = self.Q_eval.loss(state_action_values, state_action_values_target.detach())
                
                # 记录损失值
                loss_value = loss.item()
                self.total_loss += loss_value
                self.batch_count += 1
                
                logger.info(f"Batch Loss [{self.__class__.__name__}]: {loss_value:.6f}, Average Loss: {self.get_average_loss():.6f}")
                
                loss.backward()
                self.Q_eval.optimizer.step()

            # 更新epsilon值
            if self.steps > 400000:
                if self.EPSILON > self.EPS_END:
                    self.EPSILON = self.EPSILON * 0.99
                else:
                    self.EPSILON = self.EPS_END

            self.learn_step_counter = self.learn_step_counter + 1
            
            return loss_value

    def reset_counters(self):
        self.action_count = 0
        self.forward_time = 0
        self.learn_count = 0
        self.learn_time = 0
        # 重置损失计数器
        self.total_loss = 0.0
        self.batch_count = 0
        
    def get_average_loss(self):
        """获取当前epoch的平均损失值"""
        if self.batch_count > 0:
            return self.total_loss / self.batch_count
        return 0.0

class Agent_C(BaseAgent):
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, batch_size, action_space_size, input_dim, is_gpu, epsEnd=0.1, replace=20):
        super().__init__(gamma, epsilon, alpha, maxMemorySize, batch_size, action_space_size, input_dim, is_gpu, epsEnd, replace)
    
class Agent_R(BaseAgent):
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, batch_size, action_space_size, input_dim, is_gpu, epsEnd=0.1, replace=20):
        super().__init__(gamma, epsilon, alpha, maxMemorySize, batch_size, action_space_size, input_dim, is_gpu, epsEnd, replace)
    
    def chooseAction(self, observation):
        """Agent-R专用的动作选择方法，使用2维特征而不是4维"""
        with phase_timer(f"{self.__class__.__name__} Phase 3: Action Selection"):
            start = time.perf_counter()
            # 确保observation是在GPU上的tensor
            if isinstance(observation, np.ndarray):
                observation = torch.FloatTensor(observation).to(device)
            else:
                observation = observation.to(device)
            
            # Agent-R使用2维特征：(Δmi, mi)
            expected_dim = 2 * self.action_space_size
            
            # 如果是一维向量，确保长度正确
            if observation.dim() == 1:
                if observation.size(0) != expected_dim:
                    print(f"警告 [Agent-R]: 输入向量维度 {observation.size(0)} 与期望维度 {expected_dim} 不匹配")
                    # 调整向量长度
                    if observation.size(0) < expected_dim:
                        # 不足则补零
                        padded = torch.zeros(expected_dim, device=device)
                        padded[:observation.size(0)] = observation
                        observation = padded
                    else:
                        # 过多则截断
                        observation = observation[:expected_dim]
                
                # 添加批次维度
                observation = observation.unsqueeze(0)
            
            logger.debug(f"Agent-R状态向量形状: {observation.shape}")
                
            rand = np.random.random()
            
            # 使用混合精度推理 - 修改判断条件，确保CUDA可用
            if torch.cuda.is_available() and self.Q_eval.scaler is not None:
                with amp.autocast():
                    actions = self.Q_eval.forward(observation)
            else:
                actions = self.Q_eval.forward(observation)
                
            if rand < 1 - self.EPSILON:
                # action = torch.argmax(actions).item()
                # 修正这里，确保获取到标量或单元素张量
                if actions.dim() > 1:
                    # 如果actions是二维张量，先取最大值所在的列索引
                    _, action = torch.max(actions, 1)  # 沿第1维取最大值
                else:
                    # 如果actions是一维张量，直接取最大值所在的索引
                    _, action = torch.max(actions[:self.action_space_size], 0)
                # 确保action是标量
                if action.numel() == 1:  # 如果只有一个元素
                    action_value = action.item()  # 将张量转换为Python标量
                else:
                    # 如果有多个元素（这种情况不应该发生，但为了健壮性）
                    action_value = action[0].item()  # 取第一个元素
            else:
                # action = np.random.choice(self.actionSpace)
                action = randint(0, self.action_space_size - 1)
                action_value = action  # 已经是Python标量
            self.steps = self.steps + 1
            self.forward_time += time.perf_counter() - start  # 累计前向传播时间
            self.action_count += 1  # 增加动作计数
            return action_value  # 直接返回Python标量

def get_process_memory():
    """
    获取当前进程的内存使用情况
    
    Returns:
        float: 当前进程使用的内存大小（MB）
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

import os
import matplotlib.pyplot as plt
import igraph as ig

def plot_graph_comparison(original_graph, simplified_graph, title, save_path=None):
    """
    绘制原始图和简化图的比较（igraph版本）。
    如果无法使用matplotlib，会生成文本文件描述图的节点和边信息。
    参数:
        original_graph: igraph.Graph（vs["name"]为原编号）
        simplified_graph: igraph.Graph（vs["name"]为原编号）
        title: 图标题
        save_path: png保存路径或文本保存路径
    """
    is_png_save = save_path and save_path.lower().endswith('.png')
    if save_path:
        try:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"创建目录: {directory}")
        except Exception as e:
            print(f"设置保存目录时出错: {e}")

    # 保存文本描述（节点、边及统计信息）
    if save_path:
        text_path = save_path + ".txt" if is_png_save else save_path
        try:
            with open(text_path, 'w') as f:
                f.write(f"{title}\n\n")
                f.write(f"原始图: {original_graph.vcount()} 节点, {original_graph.ecount()} 边\n")
                f.write(f"简化图: {simplified_graph.vcount()} 节点, {simplified_graph.ecount()} 边\n\n")
                f.write("原始图节点: " + str(sorted(list(original_graph.vs["name"]))) + "\n\n")
                f.write("简化图节点: " + str(sorted(list(simplified_graph.vs["name"]))) + "\n\n")
                # 统计
                f.write("== 统计信息 ==\n")
                f.write(f"删除的节点数: {original_graph.vcount() - simplified_graph.vcount()}\n")
                f.write(f"删除的边数: {original_graph.ecount() - simplified_graph.ecount()}\n")
            print(f"成功保存图形文本描述到: {text_path}")
        except Exception as e:
            print(f"保存文本描述时出错: {e}")

    # 可视化部分
    try:
        plt.clf()
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')

        print(f"绘制原始图 ({original_graph.vcount()} 节点, {original_graph.ecount()} 边)...")

        # igraph布局和绘图（原始图）
        layout_orig = original_graph.layout("fr", seed=42)
        node_names_orig = [str(name) for name in original_graph.vs["name"]]
        node_size_orig = max(500 / (original_graph.vcount() ** 0.5), 20)
        ig.plot(
            original_graph,
            target=ax1,
            layout=layout_orig,
            vertex_color="#5a7dc8",
            vertex_size=node_size_orig,
            vertex_label=node_names_orig if original_graph.vcount() < 50 else None,
            vertex_label_size=10 if original_graph.vcount() < 50 else 0,
            edge_color="gray",
            edge_width=0.2,
            edge_curved=False,
            bbox=(500, 500),
            margin=50,
        )
        ax1.set_title(f'Original Graph\n({original_graph.vcount()} nodes, {original_graph.ecount()} edges)',
                      fontsize=14, fontweight='bold')
        ax1.set_axis_off()

        print(f"绘制简化图 ({simplified_graph.vcount()} 节点, {simplified_graph.ecount()} 边)...")
        layout_simp = simplified_graph.layout("fr", seed=42)
        node_names_simp = [str(name) for name in simplified_graph.vs["name"]]
        node_size_simp = max(800 / (simplified_graph.vcount() ** 0.5), 30)
        ig.plot(
            simplified_graph,
            target=ax2,
            layout=layout_simp,
            vertex_color="#4daf4a",
            vertex_size=node_size_simp,
            vertex_label=node_names_simp if simplified_graph.vcount() < 50 else None,
            vertex_label_size=12 if simplified_graph.vcount() < 50 else 0,
            edge_color="#404040",
            edge_width=0.3,
            edge_curved=False,
            bbox=(500, 500),
            margin=50,
        )
        ax2.set_title(f'Simplified Graph\n({simplified_graph.vcount()} nodes, {simplified_graph.ecount()} edges)',
                      fontsize=14, fontweight='bold')
        ax2.set_axis_off()

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(pad=2.0)

        if save_path and is_png_save:
            try:
                print(f"正在保存图像到: {save_path}")
                plt.savefig(save_path, format='png')
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    print(f"图像已成功保存到: {save_path} (大小: {file_size} 字节)")
                else:
                    print(f"警告: 图像保存失败 - 文件不存在: {save_path}")
            except Exception as e:
                print(f"保存图像时出错: {str(e)}")
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"显示图像时出错: {e}")

        try:
            plt.gcf().canvas.flush_events()
        except Exception as e:
            print(f"刷新画布时出错: {e}")

        plt.close(fig)
        print("绘图完成")
    except ImportError:
        print("matplotlib模块不可用，无法生成可视化图像，已保存文本描述")
    except Exception as e:
        print(f"绘图过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

def highRL(G_train, original_energy, energy_ratio, dataset_name, train=False, training_epoch=1000, 
           action_space_size=5, alpha=0.003,
           save_model_interval=1000, target_update=1000, run_inference=True,
           resume_checkpoint=None, checkpoint_dir="checkpoints", 
           use_amp=True, num_workers=8, batch_size=256, k2=None,
           test_interval=1000, num_test_runs=10, G_test=None, test_original_energy=None,
           start_epoch=0, train_start_node=None, name_to_idx=None,
           collection_mode: str = "greedy", window_size: int = None):
    
    global phase_stats

    # ...原始头部参数检查、agent初始化、AMP配置等保留
    start_time = time.time()
    # 随机种子改由调用方在外部设置（highest_run.py）或在此文件其他入口处设置
    # 初始化training_stats，确保无论是否训练都有定义
    training_stats = {
        "collected_rewards": [],
        "removed_nodes": [],
        "epoch_times": [],
        "loss_C": [],
        "loss_R": [],
        "epsilon": []
    }

    # 创建当前图的 name_to_idx，以避免索引失效
    current_name_to_idx = {v['name']: v.index for v in G_train.vs}
    
    actual_k2 = k2 if k2 is not None else action_space_size
    print(f"Agent-R动作空间大小设置: k2={actual_k2}")
    print(f"Agent-C收集模式: {collection_mode}")
    if collection_mode == "greedy_window":
        print(f"窗口大小: {window_size}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"当前设备: {device}")
    if device.type == 'cuda':
        print(f"GPU内存状态: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.max_memory_allocated()/1024**2:.2f}MB (当前/峰值)")
    
    k1_fixed = action_space_size  # 使用action_space_size作为固定的k1值
    input_dim_C = k1_fixed * 4  # Agent-C: 4维特征 × k1个候选节点
    input_dim_R = actual_k2 * 2  # Agent-R: 2维特征 × k2个节点动作 (Δm, m)
    
    print(f"神经网络输入维度 - Agent-C: {input_dim_C} (4维特征 × {k1_fixed}个节点)")
    print(f"神经网络输入维度 - Agent-R: {input_dim_R} (2维特征 × {actual_k2}个动作)")
    
    # 初始化两个智能体，使用不同的输入维度
    agent_C = Agent_C(gamma=0.99, epsilon=0.9, alpha=0.003, maxMemorySize=10000, batch_size=batch_size, 
                    action_space_size=action_space_size, input_dim=input_dim_C, is_gpu=True)
    agent_R = Agent_R(gamma=0.99, epsilon=0.9, alpha=0.003, maxMemorySize=10000, batch_size=batch_size, 
                    action_space_size=actual_k2, input_dim=input_dim_R, is_gpu=True)
    
    # 明确将模型移动到设备上
    agent_C.Q_eval.to(device)
    agent_C.Q_target.to(device)
    agent_R.Q_eval.to(device)
    agent_R.Q_target.to(device)
    
    # 确保混合精度训练被启用 - 统一为Q_eval和Q_target都设置scaler
    if torch.cuda.is_available() and use_amp:
        agent_C.Q_eval.scaler = torch.cuda.amp.GradScaler()
        agent_C.Q_target.scaler = torch.cuda.amp.GradScaler()
        agent_R.Q_eval.scaler = torch.cuda.amp.GradScaler()
        agent_R.Q_target.scaler = torch.cuda.amp.GradScaler()
        print(f"已手动启用混合精度训练，可加速GPU计算并减少内存使用")
    else:
        # 如果禁用AMP，确保scaler是None
        agent_C.Q_eval.scaler = None
        agent_C.Q_target.scaler = None
        agent_R.Q_eval.scaler = None
        agent_R.Q_target.scaler = None
        print("已禁用混合精度训练，使用全精度模式")
    
    print(f"模型参数数量 - Agent_C: {sum(p.numel() for p in agent_C.Q_eval.parameters())}")
    print(f"模型参数数量 - Agent_R: {sum(p.numel() for p in agent_R.Q_eval.parameters())}")
    
    # 配置多进程数据处理
    import torch.multiprocessing as mp
    if num_workers > 0:
        mp.set_start_method('spawn', force=True)  # 确保多进程的正确启动
        print(f"启用多进程数据处理，工作进程数: {num_workers}")
    
    # 添加损失跟踪列表
    loss_history_C = []
    loss_history_R = []
    epsilon_history = []
    
    # 训练阶段
    if train:
        # 重置phase_stats以确保从零开始累计
        for key in phase_stats:
            phase_stats[key] = {"count": 0, "total_time": 0}

        # 检查是否启用混合精度训练 - 修改判断条件，确保CUDA可用
        amp_enabled = torch.cuda.is_available() and hasattr(agent_C.Q_eval, 'scaler') and agent_C.Q_eval.scaler is not None
        if amp_enabled:
            print("已启用自动混合精度训练(AMP)，可以减少约50%的显存使用并加速训练")
        else:
            print("警告: 未启用自动混合精度训练(AMP)，原因可能是GPU不可用或其他问题")
            
        start_epoch = 0

        if resume_checkpoint:
            checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_{resume_checkpoint}_checkpoint.pt")
            if os.path.exists(checkpoint_path):
                try:
                    print(f"尝试从检查点恢复训练: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    # 恢复模型参数
                    agent_C.Q_eval.load_state_dict(checkpoint['agent_C_state_dict'])
                    agent_R.Q_eval.load_state_dict(checkpoint['agent_R_state_dict'])
                    agent_C.Q_target.load_state_dict(checkpoint['agent_C_target_state_dict'])
                    agent_R.Q_target.load_state_dict(checkpoint['agent_R_target_state_dict'])
                    
                    # 恢复优化器状态
                    agent_C.Q_eval.optimizer.load_state_dict(checkpoint['agent_C_optimizer'])
                    agent_R.Q_eval.optimizer.load_state_dict(checkpoint['agent_R_optimizer'])
                    
                    # 恢复Agent状态
                    agent_C.EPSILON = checkpoint['agent_C_epsilon']
                    agent_R.EPSILON = checkpoint['agent_R_epsilon']
                    agent_C.steps = checkpoint['agent_C_steps']
                    agent_R.steps = checkpoint['agent_R_steps']
                    agent_C.learn_step_counter = checkpoint['agent_C_learn_counter']
                    agent_R.learn_step_counter = checkpoint['agent_R_learn_counter']
                    
                    # 恢复经验回放缓冲区
                    if 'agent_C_memory' in checkpoint and 'agent_R_memory' in checkpoint:
                        agent_C.memory = checkpoint['agent_C_memory']
                        agent_R.memory = checkpoint['agent_R_memory']
                        agent_C.memCntr = checkpoint['agent_C_memCntr']
                        agent_R.memCntr = checkpoint['agent_R_memCntr']
                        print(f"已恢复经验回放缓冲区 - Agent_C: {len(agent_C.memory)}个样本, Agent_R: {len(agent_R.memory)}个样本")
                    else:
                        print("检查点中不包含经验回放缓冲区数据，使用空缓冲区")
                    
                    # 恢复训练进度
                    start_epoch = checkpoint['epoch'] + 1
                    training_stats = checkpoint['training_stats']
                    
                    # 恢复随机数生成器状态
                    torch.set_rng_state(checkpoint['torch_rng_state'])
                    np.random.set_state(checkpoint['numpy_rng_state'])
                    
                    print(f"成功从epoch {start_epoch-1}恢复训练状态")
                    
                except Exception as e:
                    print(f"加载检查点出错: {e}")
                    print("将从头开始训练")
                    start_epoch = 0
            else:
                print(f"检查点文件{checkpoint_path}不存在，将从头开始训练")
        
        print(f"\n=== 训练启动 | 数据集: {dataset_name} | 从epoch {start_epoch}开始，共{training_epoch}轮 ===")
        # 1. 映射原编号到索引，后续子图/邻居等统一用
        # name_to_idx = {name: idx for idx, name in enumerate(G_train.vs["name"])}
        idx_to_name = {v.index: v['name'] for v in G_train.vs}

        # 2. 获取所有连通分量（原编号集合）
        connected_components = [set(g.vs["name"]) for g in G_train.connected_components().subgraphs()]
        num_components = len(connected_components)
        print(f"总连通分量数: {num_components}")

        # ...原始内存检查、energy等统计保留
        # 获取程序初始的内存使用量（以MB为单位）并存储
        initial_memory = get_process_memory()
        # 打印初始内存使用情况，保留小数点后两位
        print(f"Initial memory usage: {initial_memory:.2f} MB")  # 输出格式如: "Initial memory usage: 123.45 MB"
        
   
        # 验证传入的原始能量参数
        if original_energy <= 0:
            raise ValueError(f"原始能量必须为正数，但得到: {original_energy}")
        print(f"\n使用预计算的原始拉普拉斯能量: {original_energy:.1f}")
        
        # 🎯 新增：计算预设能量阈值
        energy_threshold = original_energy * energy_ratio
        print(f"预设能量阈值: {energy_threshold:.1f} = {original_energy:.1f} × {energy_ratio}")
        print(f"算法目标: 让简化图拉普拉斯能量达到 {energy_threshold:.1f} (原图 {energy_ratio*100:.0f}% 的能量)")
        
        for epo in range(start_epoch, training_epoch):
            # ...epoch开始部分保留
            epoch_start_time = time.time()
            interaction_count = 0
            total_reward_R = 0
            
            # 新逻辑：初始化候选集合和核心图节点集合
            candidate_set = set()  # Agent-C收集的候选节点
            core_graph_nodes = set()  # Agent-R选择的核心图节点
            
            for idx, component in enumerate(connected_components):
                with timer(f"Component {idx+1} Processing"):
                    # 3. 获取子图（component为原编号集合，需转索引）
                    subgraph = G_train.subgraph([name_to_idx[n] for n in component])
                    # 3.1 预计算子图缓存以加速后续计算
                    degrees_cache = subgraph.degree()
                    neighbors_idx_cache = [set(subgraph.neighbors(i)) for i in range(subgraph.vcount())]
                    try:
                        clustering_cache = subgraph.transitivity_local_undirected(vertices=None)
                    except Exception:
                        clustering_cache = None
                    total_nodes = subgraph.vcount()

                    # 4. 针对子图的能量和阈值计算
                    original_energy_component = calculate_laplacian_energy(subgraph, set(subgraph.vs["name"]),name_to_idx)
                    energy_threshold_component = original_energy_component * energy_ratio

                    # 5. 初始化起始节点和集合（全为原编号）
                    available_nodes = list(subgraph.vs["name"])
                    if train_start_node and train_start_node in available_nodes:
                        start_node = train_start_node
                    else:
                        start_node = sorted(available_nodes)[0]
                        if train_start_node and epo == start_epoch:
                            print(f"警告: 指定的训练起始节点 {train_start_node} 不在图中。将使用节点 {start_node} 作为替代。")

                    candidate_set = set()
                    core_graph_nodes = {start_node}

                    # 6. 计算起始neighbors_set（neighbors需用索引转原编号）
                    start_idx = current_name_to_idx.get(start_node, -1)
                    if start_idx == -1 or start_idx >= subgraph.vcount() or start_idx < 0:
                        # 如果无效，选择第一个可用节点
                        if subgraph.vcount() > 0:
                            start_idx = 0
                            print(f"警告: 起始节点 {start_node} 无效，使用节点 {subgraph.vs[start_idx]['name']} 作为替代。")
                        else:
                            raise ValueError("图为空，无法选择起始节点。")
                    neighbors_set = set(subgraph.vs[n]["name"] for n in subgraph.neighbors(start_idx)) - core_graph_nodes
                    # 初始化窗口（仅窗口模式）
                    window_set = None
                    if collection_mode == "greedy_window":
                        if window_size is not None and window_size > 0:
                            sub_name_to_idx = {v['name']: v.index for v in subgraph.vs}
                            if degrees_cache is not None:
                                window_set = set(heapq.nlargest(
                                    min(window_size, len(neighbors_set)),
                                    list(neighbors_set),
                                    key=lambda n: degrees_cache[sub_name_to_idx.get(n, -1)] if sub_name_to_idx.get(n, -1) != -1 else -1
                                ))
                            else:
                                window_set = set(sorted(
                                    list(neighbors_set),
                                    key=lambda n: subgraph.degree(sub_name_to_idx.get(n, -1)),
                                    reverse=True
                                )[:window_size])
                    
                    initial_energy = calculate_laplacian_energy(subgraph, core_graph_nodes,name_to_idx)
                    initial_avg_node_energy = initial_energy / len(core_graph_nodes) if core_graph_nodes else 0.1
                    
                    # 初始化交互计数
                    interaction_count = 0
                    
                    while True:  # 主交互循环
                        with timer("Complete Interaction Cycle"):
                            interaction_count += 1

                            # === Agent-C 收集阶段 ===
                            with timer("Agent-C Collection Phase"):
                                print(f"\n------- 交互 {interaction_count} -------")
                                print("\n=== Agent-C 开始收集阶段 ===")

                                if not neighbors_set:
                                    print("没有可用的邻居节点，Agent-C收集阶段结束，训练将继续到下一个组件")
                                    break

                                # neighbors_set是原编号集合，subgraph是igraph子图
                                action_space_C = build_collection_action_space_new(
                                    neighbors_set, subgraph, name_to_idx, action_space_size,
                                    degrees_cache=degrees_cache, greedy=True,
                                    collection_mode=collection_mode, window_set=window_set, window_size=window_size
                                )

                                if not action_space_C or all(len(action) == 0 for action in action_space_C):
                                    print("所有动作都为空，Agent-C收集阶段结束，训练将继续到下一个组件")
                                    break

                                # Agent-C执行单次选择
                                state_C = build_state_vector_C(
                                    subgraph,
                                    action_space_C,
                                    core_graph_nodes,  # 也是原编号
                                    action_space_size,
                                    name_to_idx,
                                    degrees_cache=degrees_cache,
                                    neighbors_idx_cache=neighbors_idx_cache,
                                    clustering_cache=clustering_cache,
                                    verbose=False
                                )
                                action_C = agent_C.chooseAction(state_C)
                                chosen_nodes_by_C = action_space_C[action_C]

                                print(f"Agent-C选择的候选节点组: {chosen_nodes_by_C}")

                                # 更新候选集合为Agent-C当前选择的节点
                                candidate_set = set(chosen_nodes_by_C)
                                print(f"当前候选集合: {list(candidate_set)}")

                                # === Agent-R 选择阶段 ===
                                with timer("Agent-R Selection Phase"):
                                    print("\n=== Agent-R 开始选择阶段 ===")

                                    # 从候选集合中选择节点加入核心图
                                    available_candidates = candidate_set - core_graph_nodes

                                    if not available_candidates:
                                        print("没有可选择的候选节点，Agent-R选择阶段结束")
                                        break

                                    action_space_R, valid_deltas_R, valid_sizes_R, rejected_nodes_batch = build_selection_action_space_multi(
                                        available_candidates, core_graph_nodes, subgraph, action_space_size, k2=actual_k2, name_to_idx=name_to_idx,
                                        degrees_cache=degrees_cache, neighbors_idx_cache=neighbors_idx_cache
                                    )

                                    print(f"构建的选择动作空间: {[f'S_{size}({len(action)})' for action, size in zip(action_space_R, valid_sizes_R)]}")
                                    print(f"对应的总能量增益: {[f'{delta:.3f}' for delta in valid_deltas_R]}")
                                    print(f"对应的集合大小: {valid_sizes_R}")

                                    state_R = build_state_vector_R_multi(
                                        subgraph, action_space_R, valid_deltas_R, valid_sizes_R, core_graph_nodes, action_space_size, k2=actual_k2
                                    )
                                    print(f"状态向量维度: {len(state_R)}")

                                    action_R = agent_R.chooseAction(state_R)

                                    # 边界检查
                                    if action_R >= len(action_space_R):
                                        print(f"动作索引{action_R}超出范围({len(action_space_R)})，使用第一个动作")
                                        action_R = 0

                                    nodes_to_select = action_space_R[action_R]
                                    selected_size = valid_sizes_R[action_R]
                                    selected_total_gain = valid_deltas_R[action_R]

                                    print(f"\n🤖 Agent-R选择结果:")
                                    print(f"   选择的动作索引: {action_R}")
                                    print(f"   选择的节点集合S_{selected_size}: {sorted(nodes_to_select)}")
                                    print(f"   集合大小: {len(nodes_to_select)}")
                                    print(f"   对应的总能量增益: {selected_total_gain:.3f}")
                                    print(f"-" * 40)

                                # ===== 共享奖励与学习 =====
                                selected_nodes_set = set(nodes_to_select)
                                reward = -len(selected_nodes_set)

                                if selected_nodes_set:
                                    total_reward_R += reward
                                    print(f"奖励计算 (共享): reward = {reward} (添加了{len(selected_nodes_set)}个节点)")
                                else:
                                    print("奖励计算 (共享): reward = 0 (没有添加任何节点)")

                                with phase_timer("Graph Update: Core Nodes"):
                                    core_graph_nodes.update(selected_nodes_set)

                                # 始终使用当前子图的 name_to_idx
                                current_name_to_idx = {v['name']: v.index for v in subgraph.vs}

                                with phase_timer("Update: Neighbor Expansion"):
                                    neighbors_set -= selected_nodes_set
                                    new_neighbors = set()
                                    for selected_node in selected_nodes_set:
                                        # 使用当前映射获取 idx
                                        idx = current_name_to_idx.get(selected_node)
                                        if idx is None:
                                            raise ValueError(f"节点 {selected_node} 不存在于当前子图中")
                                        new_neighbors.update(subgraph.vs[n]["name"] for n in subgraph.neighbors(idx))
                                    neighbors_set.update(new_neighbors - core_graph_nodes)
                                    # 维护窗口（窗口模式：选取neighbors_set中度数最高的window_size个）
                                    if collection_mode == "greedy_window":
                                        if window_size is not None and window_size > 0:
                                            sub_name_to_idx = {v['name']: v.index for v in subgraph.vs}
                                            if degrees_cache is not None:
                                                window_set = set(heapq.nlargest(
                                                    min(window_size, len(neighbors_set)),
                                                    list(neighbors_set),
                                                    key=lambda n: degrees_cache[sub_name_to_idx.get(n, -1)] if sub_name_to_idx.get(n, -1) != -1 else -1
                                                ))
                                            else:
                                                window_set = set(sorted(
                                                    list(neighbors_set),
                                                    key=lambda n: subgraph.degree(sub_name_to_idx.get(n, -1)),
                                                    reverse=True
                                                )[:window_size])

                                print(f"增量更新 neighbors_set: 当前大小 {len(neighbors_set)}")
                                
                                # 计算选择后的能量状态
                                next_energy = calculate_laplacian_energy(G_train, core_graph_nodes,name_to_idx)
                                print(f"选择节点集合后，当前能量: {next_energy:.1f} (阈值: {energy_threshold_component:.1f})")

                                # ========== 核心图演变追踪 ==========
                                print(f"\n🔄 核心图演变追踪:")
                                print(f"   当前核心图节点数: {len(core_graph_nodes)}")
                                print(f"   当前能量: {next_energy:.1f}")
                                print(f"=" * 50)

                                # ===== 构建下一状态 S' =====
                                next_action_space_C = build_collection_action_space_new(
                                    neighbors_set, subgraph, name_to_idx, action_space_size,
                                    degrees_cache=degrees_cache, greedy=True,
                                    collection_mode=collection_mode, window_set=window_set, window_size=window_size
                                )
                                with phase_timer("Agent-C Phase 4: State Vector Construction (Post-action)"):
                                    next_state_C = build_state_vector_C(
                                        subgraph, next_action_space_C, core_graph_nodes,
                                        action_space_size, name_to_idx,
                                        degrees_cache=degrees_cache,
                                        neighbors_idx_cache=neighbors_idx_cache,
                                        clustering_cache=clustering_cache,
                                        verbose=False
                                    )

                                # 为Agent-R构建下一状态（基于下一轮C的最优选择）
                                next_chosen_nodes_by_C = next_action_space_C[0] if next_action_space_C and next_action_space_C[0] else []
                                with phase_timer("Agent-R Phase 4: State Vector Construction (Post-action)"):
                                    next_action_space_R, next_valid_deltas_R, next_valid_sizes_R, _ = build_selection_action_space_multi(
                                        next_chosen_nodes_by_C, core_graph_nodes, subgraph, action_space_size, k2=actual_k2, name_to_idx=name_to_idx,
                                        degrees_cache=degrees_cache, neighbors_idx_cache=neighbors_idx_cache
                                    )
                                    next_state_R = build_state_vector_R_multi(subgraph, next_action_space_R, next_valid_deltas_R, next_valid_sizes_R, core_graph_nodes, action_space_size, k2=actual_k2)

                                # 存储两个智能体的经验
                                agent_C.storeTransition(state_C, action_C, reward, next_state_C)
                                agent_R.storeTransition(state_R, action_R, reward, next_state_R)
                                
                                # 两个智能体并行学习
                                if agent_C.memCntr > batch_size:
                                    loss_c = agent_C.learn()
                                    if loss_c is not None:
                                        print(f"Batch Loss [Agent_C]: {loss_c:.6f}")
                                
                                if agent_R.memCntr > batch_size:
                                    loss_r = agent_R.learn()
                                    if loss_r is not None:
                                        print(f"Batch Loss [Agent_R]: {loss_r:.6f}")
                            
                        # 检查能量阈值
                        if len(core_graph_nodes) > 0:
                            current_energy = calculate_laplacian_energy(G_train, core_graph_nodes,name_to_idx)
                            if current_energy >= energy_threshold_component:
                                print(f"当前能量 {current_energy:.1f} 已达到组件阈值 {energy_threshold_component:.1f}，完成当前组件")
                                break


                        # 停止条件相关
                        available_nodes = set(subgraph.vs["name"]) - core_graph_nodes
                        if not available_nodes:
                            print("处理完成：所有节点都已加入核心图")
                            break
                        
                        # 检查能量比率是否达到目标
                        if len(core_graph_nodes) > 0 and original_energy_component > 0:
                            current_energy = calculate_laplacian_energy(G_train, core_graph_nodes,name_to_idx)
                            current_energy_ratio = current_energy / original_energy_component
                            if current_energy_ratio >= energy_ratio:
                                print(f"能量比率{current_energy_ratio:.3f}已达到目标阈值{energy_ratio:.3f}，完成图简化")
                                break
                        
                        # 输出本轮交互后的状态
                        print(f"\n本轮交互后的状态:")
                        print(f"候选集合({len(candidate_set)}个节点): {list(candidate_set)[:10]}{'...' if len(candidate_set) > 10 else ''}")
                        print(f"核心图节点({len(core_graph_nodes)}个节点): {list(core_graph_nodes)[:10]}{'...' if len(core_graph_nodes) > 10 else ''}")

                        # 最终核心图/子图
                    final_energy_component = calculate_laplacian_energy(subgraph, core_graph_nodes,name_to_idx)

                    print(f"\n🏆 训练最终核心图状态:")
                    print(f"   训练最终核心图节点: {sorted(core_graph_nodes)}")
                    print(f"   训练最终节点数量: {len(core_graph_nodes)}")
                    print(f"   训练最终能量: {final_energy_component:.1f}")
                    print(f"   训练目标阈值: {original_energy_component:.1f}乘{energy_ratio}={energy_threshold_component:.1f}")
                    print(f"   训练节点保留率: {len(core_graph_nodes)}/{subgraph.vcount()}={len(core_graph_nodes)/subgraph.vcount():.2%}")
                    if original_energy_component > 0:
                        print(f"   训练能量保留率: {final_energy_component/original_energy_component:.2%}")
                    else:
                        print(f"   训练能量保留率: N/A")
                    print(f"============================================================")

                # 计算epoch处理时间
                epoch_time = time.time() - epoch_start_time
                
                # 获取并存储平均损失
                avg_loss_C = agent_C.get_average_loss()
                avg_loss_R = agent_R.get_average_loss()
                loss_history_C.append(avg_loss_C)
                loss_history_R.append(avg_loss_R)
                epsilon_history.append(agent_C.EPSILON)
                
                training_stats["loss_C"].append(avg_loss_C)
                training_stats["loss_R"].append(avg_loss_R)
                training_stats["epsilon"].append(agent_C.EPSILON)
                
                # 连通分量循环结束后，打印epoch统计信息
                print(f"\nEpoch {epo + 1}/{training_epoch} 完成: " + 
                      f"交互次数: {interaction_count}, " +
                      f"Agent_C平均损失: {avg_loss_C:.6f}, " +
                      f"Agent_R平均损失: {avg_loss_R:.6f}, " +
                      f"探索率: {agent_C.EPSILON:.4f}, " +
                      f"Agent_R总奖励: {total_reward_R:.4f}, " +
                      f"本轮用时: {epoch_time:.2f}秒")
                
                print(f"最终候选集合({len(candidate_set)}个节点): {list(candidate_set)[:10]}{'...' if len(candidate_set) > 10 else ''}")
                print(f"最终核心图节点({len(core_graph_nodes)}个节点): {list(core_graph_nodes)[:10]}{'...' if len(core_graph_nodes) > 10 else ''}")

                # 每隔一定周期保存模型和可视化损失曲线
                if (epo + 1) % save_model_interval == 0 or epo + 1 == training_epoch:
                    # 保存损失曲线
                    if len(loss_history_C) > 0:
                        plt.figure(figsize=(12, 8))
                        
                        # 绘制损失曲线
                        plt.subplot(2, 1, 1)
                        plt.plot(range(1, len(loss_history_C) + 1), loss_history_C, label='Agent_C Loss')
                        plt.plot(range(1, len(loss_history_R) + 1), loss_history_R, label='Agent_R Loss')
                        plt.xlabel('Epoch')
                        plt.ylabel('Average Loss')
                        plt.title('Training Loss')
                        plt.legend()
                        plt.grid(True)
                        
                        # 绘制探索率曲线
                        plt.subplot(2, 1, 2)
                        plt.plot(range(1, len(epsilon_history) + 1), epsilon_history)
                        plt.xlabel('Epoch')
                        plt.ylabel('Epsilon')
                        plt.title('Exploration Rate')
                        plt.grid(True)
                        
                        plt.tight_layout()
                        
                        # 保存图像
                        loss_plot_path = os.path.join(checkpoint_dir, f"{dataset_name}_loss_plot_{epo+1}.png")
                        plt.savefig(loss_plot_path)
                        plt.close()
                        print(f"损失曲线已保存到: {loss_plot_path}")

                    # 创建并保存完整检查点
                    checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_{epo+1}_checkpoint.pt")
                    checkpoint = {
                        # 模型参数
                        'agent_C_state_dict': agent_C.Q_eval.state_dict(),
                        'agent_R_state_dict': agent_R.Q_eval.state_dict(),
                        'agent_C_target_state_dict': agent_C.Q_target.state_dict(),
                        'agent_R_target_state_dict': agent_R.Q_target.state_dict(),
                        
                        # 优化器状态
                        'agent_C_optimizer': agent_C.Q_eval.optimizer.state_dict(),
                        'agent_R_optimizer': agent_R.Q_eval.optimizer.state_dict(),
                        
                        # Agent状态
                        'agent_C_epsilon': agent_C.EPSILON,
                        'agent_R_epsilon': agent_R.EPSILON,
                        'agent_C_steps': agent_C.steps,
                        'agent_R_steps': agent_R.steps,
                        'agent_C_learn_counter': agent_C.learn_step_counter,
                        'agent_R_learn_counter': agent_R.learn_step_counter,
                        
                        # 经验回放缓冲区
                        'agent_C_memory': agent_C.memory,
                        'agent_R_memory': agent_R.memory, 
                        'agent_C_memCntr': agent_C.memCntr,
                        'agent_R_memCntr': agent_R.memCntr,
                        
                        # 随机数生成器状态
                        'torch_rng_state': torch.get_rng_state(),
                        'numpy_rng_state': np.random.get_state(),
                        
                        # 训练进度
                        'epoch': epo,
                        'training_stats': training_stats,
                        
                        # 训练参数
                        'action_space_size': action_space_size,
                        'batch_size': agent_C.batch_size,
                        'memory_size': agent_C.memSize
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"检查点已保存到: {checkpoint_path}")
                    
                    # 为了兼容性，也保存旧格式的模型文件
                    torch.save(agent_C.Q_eval.state_dict(), os.path.join(checkpoint_dir, f"agent_C_checkpoint_{epo+1}.pth"))
                    torch.save(agent_R.Q_eval.state_dict(), os.path.join(checkpoint_dir, f"agent_R_checkpoint_{epo+1}.pth"))
                    print(f"兼容格式模型已保存 (epoch {epo+1})")

                # 更新目标网络
                if (epo + 1) % target_update == 0:
                    agent_C.Q_target.load_state_dict(agent_C.Q_eval.state_dict())
                    agent_R.Q_target.load_state_dict(agent_R.Q_eval.state_dict())
                    print(f"Target networks updated at epoch {epo+1}")

                # 每个epoch结束时打印性能统计
                print("\nPerformance Statistics:")
                print(f"Agent-C - Avg Forward Time: {agent_C.forward_time/max(1,agent_C.action_count):.6f}s")
                print(f"Agent-C - Avg Learn Time: {agent_C.learn_time/max(1,agent_C.learn_count):.6f}s")
                print(f"Agent-R - Avg Forward Time: {agent_R.forward_time/max(1,agent_R.action_count):.6f}s")
                print(f"Agent-R - Avg Learn Time: {agent_R.learn_time/max(1,agent_R.learn_count):.6f}s")

                # 在每个epoch结束时打印阶段统计信息
                logger.info("\n=== 阶段统计信息 ===")
                logger.info("\nAgent-C收集阶段：")
                logger.info(f"  > P1. 构建动作空间: {phase_stats['Agent-C Phase 1: Action Space Construction']['total_time']:.4f} s ({phase_stats['Agent-C Phase 1: Action Space Construction']['count']} calls)")
                logger.info(f"    - P1a. 邻居集构建: {phase_stats['Agent-C P1a: Neighbor Set Construction']['total_time']:.4f} s")
                logger.info(f"    - P1b. 候选池排序: {phase_stats['Agent-C P1b: Candidate Pool Ranking']['total_time']:.4f} s")
                logger.info(f"    - P1c. 贪心选择: {phase_stats['Agent-C P1c: Greedy Selection']['total_time']:.4f} s")
                logger.info(f"  > P2. 构建状态向量 (Pre-Action): {phase_stats['Agent-C Phase 2: State Vector Construction (Pre-action)']['total_time']:.4f} s ({phase_stats['Agent-C Phase 2: State Vector Construction (Pre-action)']['count']} calls)")
                logger.info(f"    - P2a. Δ Laplacian Energy: {phase_stats['Agent-C P2a: Delta Laplacian Energy']['total_time']:.4f} s")
                logger.info(f"    - P2b. Δ Edges: {phase_stats['Agent-C P2b: Delta Edges']['total_time']:.4f} s")
                logger.info(f"    - P2c. Δ Overlap: {phase_stats['Agent-C P2c: Delta Overlap']['total_time']:.4f} s")
                logger.info(f"    - P2d. Clustering Coeff: {phase_stats['Agent-C P2d: Clustering Coefficient']['total_time']:.4f} s")
                logger.info(f"  > P3. 选择动作: {phase_stats['Agent-C Phase 3: Action Selection']['total_time']:.4f} s ({phase_stats['Agent-C Phase 3: Action Selection']['count']} calls)")
                logger.info(f"  > P4. 构建状态向量 (Post-Action): {phase_stats['Agent-C Phase 4: State Vector Construction (Post-action)']['total_time']:.4f} s ({phase_stats['Agent-C Phase 4: State Vector Construction (Post-action)']['count']} calls)")
                logger.info(f"  > P5. 学习: {phase_stats['Agent-C Phase 5: Learning']['total_time']:.4f} s ({phase_stats['Agent-C Phase 5: Learning']['count']} calls)")

                logger.info("\nAgent-R删除阶段：")
                logger.info(f"  > P1. 构建动作空间: {phase_stats['Agent-R Phase 1: Action Space Construction']['total_time']:.4f} s ({phase_stats['Agent-R Phase 1: Action Space Construction']['count']} calls)")
                logger.info(f"  > P2. 构建状态向量 (Pre-Action): {phase_stats['Agent-R Phase 2: State Vector Construction (Pre-action)']['total_time']:.4f} s ({phase_stats['Agent-R Phase 2: State Vector Construction (Pre-action)']['count']} calls)")
                logger.info(f"  > P3. 选择动作: {phase_stats['Agent-R Phase 3: Action Selection']['total_time']:.4f} s ({phase_stats['Agent-R Phase 3: Action Selection']['count']} calls)")
                logger.info(f"  > P4. 构建状态向量 (Post-Action): {phase_stats['Agent-R Phase 4: State Vector Construction (Post-action)']['total_time']:.4f} s ({phase_stats['Agent-R Phase 4: State Vector Construction (Post-action)']['count']} calls)")
                logger.info(f"  > P5. 学习: {phase_stats['Agent-R Phase 5: Learning']['total_time']:.4f} s ({phase_stats['Agent-R Phase 5: Learning']['count']} calls)")
                logger.info(f"  > 组件能量计算: {phase_stats['Agent-R Component Energy Calculation']['total_time']:.4f} s ({phase_stats['Agent-R Component Energy Calculation']['count']} calls)")

                # 重置 phase_stats 避免累计到下一个 epoch
                # for key in phase_stats:
                #     phase_stats[key] = {"count": 0, "total_time": 0}

                # 每100个epoch打印内存使用情况
                if (epo + 1) % 100 == 0:
                    current_memory = get_process_memory()
                    print(f"Memory usage at epoch {epo+1}: {current_memory:.2f} MB")
                    print(f"Memory increase: {current_memory - initial_memory:.2f} MB")

                # 在每个epoch结束时获取平均损失
                avg_loss_C = agent_C.get_average_loss()
                avg_loss_R = agent_R.get_average_loss()
                
  
                # 重置计数器
                agent_C.reset_counters()
                agent_R.reset_counters()

            # 所有epoch结束后才打印这些信息
            print("\nRL简化完成。")
            selected_indices = [name_to_idx[n] for n in core_graph_nodes]
            simplified = subgraph.subgraph(selected_indices)
            print(f"简化后图节点总数: {simplified.vcount()}, 边数: {simplified.ecount()}")

            print(f"候选集合节点数: {len(candidate_set)}")
            total_time = time.time() - start_time
            print(f"Training completed in {total_time:.2f} seconds.")

        # 可视化整轮训练的结果
        original_component = subgraph
        final_simplified = subgraph.subgraph([name_to_idx[n] for n in core_graph_nodes])

        # 可视化/保留指标
        viz_dir = os.path.join(os.getcwd(), f"visualizations_{dataset_name}")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir, exist_ok=True)
            print(f"创建可视化目录: {viz_dir}")
        
        # 计算并输出图保留质量指标
        print("\n=== 图保留质量评估结果 ===")
        preservation_metrics = evaluate_graph_preservation(subgraph, final_simplified,name_to_idx)
        print(f"节点保留率: {preservation_metrics['node_preservation']:.4f}")
        print(f"边保留率: {preservation_metrics['edge_preservation']:.4f}")
        
        # 将评估结果保存到文件
        metrics_file = os.path.join(viz_dir, f"epoch_{epo+1}_component_{idx+1}_metrics.txt")
        try:
            with open(metrics_file, 'w') as f:
                f.write("=== 图保留质量评估结果 ===\n")
                for metric, value in preservation_metrics.items():
                    f.write(f"{metric}: {value:.6f}\n")
                
                # 添加其他基本信息
                f.write(f"\n原图节点数: {original_component.vcount()}\n")
                f.write(f"原图边数: {original_component.ecount()}\n")
                f.write(f"简化图节点数: {final_simplified.vcount()}\n")
                f.write(f"简化图边数: {final_simplified.ecount()}\n")
                f.write(f"删除的节点数: {original_component.vcount() - final_simplified.vcount()}\n")
                f.write(f"删除的边数: {original_component.ecount() - final_simplified.ecount()}\n")
                
            print(f"评估指标已保存到: {metrics_file}")
        except Exception as e:
            print(f"保存评估指标时出错: {e}")
        
        # 创建保存路径
        save_path = os.path.join(viz_dir, f"epoch_{epo+1}_component_{idx+1}_final.png")
        plot_graph_comparison(subgraph, final_simplified,
                            f'Epoch {epo+1} Final Result (Component {idx+1})',
                            save_path=save_path)
        # ...训练日志/保存 checkpoint 代码全部保留不变
# 当前保存检查点的部分
    # 保存最终检查点
    final_checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_{training_epoch}_checkpoint.pt")
    checkpoint = {
        # 模型参数
        'agent_C_state_dict': agent_C.Q_eval.state_dict(),
        'agent_R_state_dict': agent_R.Q_eval.state_dict(),
        'agent_C_target_state_dict': agent_C.Q_target.state_dict(),
        'agent_R_target_state_dict': agent_R.Q_target.state_dict(),
        
        # 优化器状态
        'agent_C_optimizer': agent_C.Q_eval.optimizer.state_dict(),
        'agent_R_optimizer': agent_R.Q_eval.optimizer.state_dict(),
        
        # Agent状态
        'agent_C_epsilon': agent_C.EPSILON,
        'agent_R_epsilon': agent_R.EPSILON,
        'agent_C_steps': agent_C.steps,
        'agent_R_steps': agent_R.steps,
        'agent_C_learn_counter': agent_C.learn_step_counter,
        'agent_R_learn_counter': agent_R.learn_step_counter,
        
        # 经验回放缓冲区
        'agent_C_memory': agent_C.memory,
        'agent_R_memory': agent_R.memory,
        'agent_C_memCntr': agent_C.memCntr,
        'agent_R_memCntr': agent_R.memCntr,
        
        # 随机数生成器状态
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        
        # 训练进度
        'epoch': training_epoch - 1,
        'training_stats': training_stats,
        
        # 训练参数 - 添加实验关键参数
        'action_space_size': action_space_size,
        'k2': actual_k2,  # Agent-R的动作空间大小
        'batch_size': batch_size,
        'memory_size': 10000,
        'energy_ratio': energy_ratio,
        'alpha': alpha
    }
    torch.save(checkpoint, final_checkpoint_path)
    print(f"最终检查点已保存到: {final_checkpoint_path}")
    
    # 保存训练统计数据到单独的文件 - 新增内容
    stats_dir = os.path.join(checkpoint_dir, "training_stats")
    os.makedirs(stats_dir, exist_ok=True)
    
    # 保存详细的训练统计数据为JSON
    stats_filename = f"{dataset_name}_C{action_space_size}_R{actual_k2}_epoch{training_epoch}_stats.json"
    stats_path = os.path.join(stats_dir, stats_filename)
    
    training_summary = {
        'experiment_config': {
            'dataset': dataset_name,
            'energy_ratio': energy_ratio,
            'training_epochs': training_epoch,
            'action_space_C': action_space_size,
            'action_space_R': actual_k2,
            'alpha': alpha,
            'batch_size': batch_size
        },
        'training_stats': training_stats,
        'final_performance': {
            'final_epsilon_C': agent_C.EPSILON,
            'final_epsilon_R': agent_R.EPSILON,
            'total_steps_C': agent_C.steps,
            'total_steps_R': agent_R.steps,
            'learn_steps_C': agent_C.learn_step_counter,
            'learn_steps_R': agent_R.learn_step_counter
        }
    }
    
    try:
        import json
        with open(stats_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        print(f"训练统计数据已保存到: {stats_path}")
    except Exception as e:
        print(f"保存训练统计数据时出错: {e}")
    
    # 保存CSV格式的训练数据便于画图
    csv_filename = f"{dataset_name}_C{action_space_size}_R{actual_k2}_epoch{training_epoch}_training.csv"
    csv_path = os.path.join(stats_dir, csv_filename)
    
    try:
        import pandas as pd
        # 创建训练数据的DataFrame
        max_len = max(len(training_stats.get('epoch_times', [])),
                     len(training_stats.get('loss_C', [])),
                     len(training_stats.get('loss_R', [])),
                     len(training_stats.get('epsilon', [])))
        
        if max_len > 0:
            df_data = {}
            for key, values in training_stats.items():
                if isinstance(values, list) and len(values) > 0:
                    # 用None填充较短的列表
                    padded_values = values + [None] * (max_len - len(values))
                    df_data[key] = padded_values
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.to_csv(csv_path, index=False)
                print(f"训练数据CSV已保存到: {csv_path}")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
    
    # 为了兼容性，也保存旧格式的模型文件
    torch.save(agent_C.Q_eval.state_dict(), os.path.join(checkpoint_dir, f"agent_C_checkpoint_{training_epoch}.pth"))
    torch.save(agent_R.Q_eval.state_dict(), os.path.join(checkpoint_dir, f"agent_R_checkpoint_{training_epoch}.pth"))
    print(f"Final compatible models saved at epoch {training_epoch}")
    print("\n==== 训练阶段结束 ====")
    
    # 推理阶段 - 根据run_inference参数决定是否执行
    if run_inference:
        print("\n==== 推理阶段开始 ====")
        # 使用训练好的模型或加载已有模型进行推理
        simplified_graph, metrics, inference_time, phase_stats = infer_highRL(
            G_train, 
            original_energy, 
            energy_ratio, 
            dataset_name,
            agent_C if train else None,  # 如果刚刚训练过，使用训练好的模型
            agent_R if train else None,
            action_space_size=action_space_size,
            model_epoch=training_epoch,
            use_amp=use_amp,  # 传递use_amp参数
            num_workers=num_workers,  # 传递num_workers参数
            batch_size=batch_size,  # 传递batch_size参数
            k2=actual_k2,  # 传递k2参数
            start_node=train_start_node  # 传入起始节点
        )
        print("\n==== 推理阶段结束 ====")
        return agent_C, agent_R, simplified_graph, metrics, inference_time, phase_stats
    
    # 当不执行推理时，返回智能体和训练统计信息以及总训练时间
    total_training_time = time.time() - start_time
    return agent_C, agent_R, training_stats, total_training_time, phase_stats

def infer_highRL(G_test, original_energy: float, energy_ratio, dataset_name, agent_C=None, agent_R=None, 
               action_space_size=5, alpha=0.003,
               model_path=None, model_epoch=10000,
               use_checkpoint=False, use_amp=True, num_workers=8, batch_size=256, k2=None, start_node=None, name_to_idx=None,
               collection_mode: str = "greedy", window_size: int = None):

    start_time = time.time()
    print("\n==== 推理阶段开始 ====")
    # 随机种子改由调用方在外部设置

    actual_k2 = k2 if k2 is not None else action_space_size
    print(f"Agent-R动作空间大小设置: k2={actual_k2}")
    print(f"Agent-C收集模式: {collection_mode}")
    if collection_mode == "greedy_window":
        print(f"窗口大小: {window_size}")

    print(f"当前设备: {device}")
    if device.type == 'cuda':
        print(f"GPU内存状态: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.max_memory_allocated()/1024**2:.2f}MB (当前/峰值)")
    # 如果没有提供agent则加载
    if agent_C is None or agent_R is None:
        k1_fixed = action_space_size
        input_dim_C = 4 * k1_fixed
        input_dim_R = actual_k2 * 2
        print(f"推理阶段神经网络输入维度 - Agent-C: {input_dim_C} (4维特征 × {k1_fixed}个候选节点)")
        print(f"推理阶段神经网络输入维度 - Agent-R: {input_dim_R} (2维特征 × {actual_k2}个动作)")

        agent_C = Agent_C(gamma=0.99, epsilon=0.0, alpha=0.003, maxMemorySize=10000,
                         batch_size=batch_size, action_space_size=action_space_size,
                         input_dim=input_dim_C, is_gpu=True)
        agent_R = Agent_R(gamma=0.99, epsilon=0.0, alpha=0.003, maxMemorySize=10000,
                         batch_size=batch_size, action_space_size=actual_k2,
                         input_dim=input_dim_R, is_gpu=True)

        model_path = model_path or os.getcwd()
        try:
            if use_checkpoint:
                checkpoint_path = os.path.join(model_path, f"{dataset_name}_{model_epoch}_checkpoint.pt")
                print(f"尝试从检查点加载模型: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                agent_C.Q_eval.load_state_dict(checkpoint['agent_C_state_dict'])
                agent_R.Q_eval.load_state_dict(checkpoint['agent_R_state_dict'])
                print("成功从检查点加载模型")
            else:
                print("使用传统方式加载模型...")
                agent_C_path = os.path.join(model_path, f"agent_C_checkpoint_{model_epoch}.pth")
                agent_R_path = os.path.join(model_path, f"agent_R_checkpoint_{model_epoch}.pth")
                try:
                    agent_C.Q_eval.load_state_dict(torch.load(agent_C_path, map_location=device))
                    agent_R.Q_eval.load_state_dict(torch.load(agent_R_path, map_location=device))
                    print("成功加载模型")
                except FileNotFoundError as e:
                    print(f"加载模型出错: {e}")
                    print("模型加载失败，将使用随机初始化的模型进行推理")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("将使用随机初始化的模型进行推理")

    agent_C.eval()
    agent_R.eval()

    all_simplified_graphs = []
    all_metrics = []

    # igraph获取连通分量
    subgraph_list = G_test.connected_components().subgraphs()
    print(f"\n图中包含 {len(subgraph_list)} 个连通分量")

    for idx, subgraph in enumerate(subgraph_list):
        component = set(subgraph.vs["name"]) # 原始编号
        print(f"\n处理连通分量 {idx+1}/{len(subgraph_list)}")
        print(f"当前连通分量: {subgraph.vcount()}节点, {subgraph.ecount()}边")

        # 建立当前子图的 name->idx 映射，便于后续索引（覆盖函数参数）
        name_to_idx = {v['name']: v.index for v in subgraph.vs}

        # 计算能量阈值
        original_energy_component = calculate_laplacian_energy(subgraph, set(subgraph.vs["name"]),name_to_idx)
        energy_threshold = original_energy_component * energy_ratio
        print(f"连通分量原始能量: {original_energy_component:.1f}")
        print(f"能量阈值({energy_ratio:.1%}): {energy_threshold:.1f}")

        core_graph_nodes = set()
        candidate_set = set()
        rejected_nodes = set()

        # 预计算子图缓存
        degrees_cache = subgraph.degree()
        neighbors_idx_cache = [set(subgraph.neighbors(i)) for i in range(subgraph.vcount())]
        try:
            clustering_cache = subgraph.transitivity_local_undirected(vertices=None)
        except Exception:
            clustering_cache = None

        # 选择起始节点（优先级：显式start_node → 数据集默认常量 → 最大度）
        if start_node and start_node in component:
            chosen_start_node = start_node
            print(f"使用指定的起始节点: {chosen_start_node}")
        else:
            # 尝试数据集默认常量
            dataset_lower = str(dataset_name).lower() if dataset_name is not None else ""
            preferred = None
            if dataset_lower.endswith("inference_only") or dataset_lower.endswith("inference_ratio_0.5"):
                # 保持通用性：仅根据前缀判断
                pass
            if dataset_lower.startswith("jazz"):
                preferred = 55
            elif dataset_lower.startswith("facebook"):
                preferred = 1912
            elif dataset_lower.startswith("engb"):
                preferred = 100
            if preferred is not None and preferred in component:
                chosen_start_node = preferred
                print(f"使用默认起始节点: {chosen_start_node}")
            else:
                # 回退为最大度节点
                degrees_cache = subgraph.degree() if 'degrees_cache' not in locals() else degrees_cache
                if len(degrees_cache) > 0:
                    max_deg_idx = int(np.argmax(degrees_cache))
                    chosen_start_node = int(subgraph.vs[max_deg_idx]['name'])
                    print(f"使用最大度起始节点: {chosen_start_node}")
                else:
                    # 最后兜底：随机
                    if start_node:
                        print(f"警告: 指定的起始节点 {start_node} 不在当前连通分量中，将随机选择。")
                    chosen_start_node = random.choice(list(component))
        core_graph_nodes.add(chosen_start_node)
        idx_start = name_to_idx[chosen_start_node]
        neighbors_set = set(subgraph.vs[n]["name"] for n in subgraph.neighbors(idx_start)) - core_graph_nodes

        print(f"初始化 - 起始节点v0: {chosen_start_node}")
        print(f"初始化 - 核心图G'只包含起始节点: {len(core_graph_nodes)}")
        print(f"初始化 - neighbors_set: {len(neighbors_set)} 个节点")

        initial_energy = calculate_laplacian_energy(subgraph, core_graph_nodes,name_to_idx)
        print(f"\n🏁 初始核心图状态:")
        print(f"   起始节点v0: {chosen_start_node}")
        print(f"   初始核心图节点: {sorted(core_graph_nodes)}")
        print(f"   初始节点数量: {len(core_graph_nodes)}")
        print(f"   初始能量: {initial_energy:.1f}")
        print(f"   目标阈值: {energy_threshold:.1f}")
        print(f"=" * 60)

        round_count = 0
        while True:
            candidate_set = neighbors_set - rejected_nodes
            if not candidate_set:
                print("没有新的候选节点，结束处理")
                break

            # 初始化/维护窗口（仅窗口模式）
            window_set = None
            if collection_mode == "greedy_window":
                if window_size is not None and window_size > 0:
                    sub_name_to_idx = {v['name']: v.index for v in subgraph.vs}
                    if degrees_cache is not None:
                        window_set = set(heapq.nlargest(
                            min(window_size, len(neighbors_set)),
                            list(neighbors_set),
                            key=lambda n: degrees_cache[sub_name_to_idx.get(n, -1)] if sub_name_to_idx.get(n, -1) != -1 else -1
                        ))
                    else:
                        window_set = set(sorted(
                            list(neighbors_set),
                            key=lambda n: subgraph.degree(sub_name_to_idx.get(n, -1)),
                            reverse=True
                        )[:window_size])

            action_space_collect = build_collection_action_space_new(
                neighbors_set, subgraph, name_to_idx, action_space_size,
                degrees_cache=degrees_cache,
                collection_mode=collection_mode, window_set=window_set, window_size=window_size
            )
            if not action_space_collect or all(len(action) == 0 for action in action_space_collect):
                print("没有可用的收集动作，结束处理")
                break

            state_collect = build_state_vector_C(
                subgraph,
                action_space_collect,
                core_graph_nodes,
                action_space_size,
                name_to_idx,
                degrees_cache=degrees_cache,
                neighbors_idx_cache=neighbors_idx_cache,
                clustering_cache=clustering_cache,
                verbose=False
            )
            action_collect = agent_C.chooseAction(state_collect)
            if action_collect >= len(action_space_collect):
                print(f"动作索引{action_collect}超出范围，使用第一个动作")
                action_collect = 0
            chosen_nodes_by_C = action_space_collect[action_collect]
            print(f"Agent-C选择的节点组: {chosen_nodes_by_C}")

            action_space_select, valid_deltas, valid_sizes, rejected_nodes_batch = build_selection_action_space_multi(
                chosen_nodes_by_C, core_graph_nodes, subgraph, action_space_size, k2=actual_k2,name_to_idx=name_to_idx
            )
            if not action_space_select:
                rejected_nodes.update(rejected_nodes_batch)
                print(f"没有满足能量增益条件的动作，将节点{list(rejected_nodes_batch)}添加到rejected_set")
                print(f"当前rejected_set大小: {len(rejected_nodes)}, 节点: {sorted(rejected_nodes)}")
                continue

            state_select = build_state_vector_R_multi(
                subgraph, action_space_select, valid_deltas, valid_sizes, core_graph_nodes, action_space_size, k2=actual_k2
            )
            action_select = agent_R.chooseAction(state_select)
            if action_select >= len(action_space_select):
                print(f"动作索引{action_select}超出范围，使用第一个动作")
                action_select = 0
            nodes_to_select = action_space_select[action_select]
            selected_size = valid_sizes[action_select]

            print(f"\n🤖 Agent-R选择结果:")
            print(f"   选择的动作索引: {action_select}")
            print(f"   选择的节点集合S_{selected_size}: {sorted(nodes_to_select)}")
            print(f"   集合大小: {len(nodes_to_select)}")
            print(f"   对应的总能量增益: {valid_deltas[action_select]:.3f}")
            print(f"-" * 40)

            selected_nodes_set = set(nodes_to_select)
            core_graph_nodes.update(selected_nodes_set)
            neighbors_set -= selected_nodes_set

            new_neighbors = set()
            for selected_node in selected_nodes_set:
                idx = name_to_idx[selected_node]
                new_neighbors.update(subgraph.vs[n]["name"] for n in subgraph.neighbors(idx))
            neighbors_set.update(new_neighbors - core_graph_nodes)
            # 维护窗口（窗口模式）
            if collection_mode == "greedy_window":
                if window_size is not None and window_size > 0:
                    sub_name_to_idx = {v['name']: v.index for v in subgraph.vs}
                    if degrees_cache is not None:
                        window_set = set(heapq.nlargest(
                            min(window_size, len(neighbors_set)),
                            list(neighbors_set),
                            key=lambda n: degrees_cache[sub_name_to_idx.get(n, -1)] if sub_name_to_idx.get(n, -1) != -1 else -1
                        ))
                    else:
                        window_set = set(sorted(
                            list(neighbors_set),
                            key=lambda n: subgraph.degree(sub_name_to_idx.get(n, -1)),
                            reverse=True
                        )[:window_size])
            print(f"增量更新 neighbors_set: 当前大小 {len(neighbors_set)}")

            round_count += 1
            current_energy = calculate_laplacian_energy(subgraph, core_graph_nodes,name_to_idx)
            print(f"\n🔄 第{round_count}轮 - 核心图演变追踪:")
            print(f"   Agent-R选择的节点集合S_{selected_size}: {sorted(nodes_to_select)}")
            print(f"   当前核心图节点: {sorted(core_graph_nodes)}")
            print(f"   核心图大小: {len(core_graph_nodes)}")
            print(f"   当前能量: {current_energy:.1f}")
            print(f"   目标阈值: {energy_threshold:.1f}")
            print(f"=" * 50)

            print(f"当前能量: {current_energy:.1f} (阈值: {energy_threshold:.1f})")
            if current_energy >= energy_threshold:
                print(f"✓ 能量条件满足: {current_energy:.1f} ≥ {energy_threshold:.1f}")
                break
            if len(core_graph_nodes) == subgraph.vcount():
                print("已包含所有节点，结束处理")
                break

        final_energy = calculate_laplacian_energy(subgraph, core_graph_nodes,name_to_idx)
        print(f"\n🏆 最终核心图状态:")
        print(f"   最终核心图节点: {sorted(core_graph_nodes)}")
        print(f"   最终节点数量: {len(core_graph_nodes)}")
        print(f"   最终能量: {final_energy:.1f}")
        print(f"   目标阈值: {energy_threshold:.1f}")
        print(f"   节点保留率: {len(core_graph_nodes)}/{subgraph.vcount()} = {len(core_graph_nodes)/subgraph.vcount():.2%}")
        print(f"   能量保留率: {final_energy/original_energy_component:.2%}")
        print(f"=" * 60)

        final_simplified = subgraph.subgraph(subgraph.vs.select(name_in=core_graph_nodes))
        all_simplified_graphs.append(final_simplified)

        try:
            metrics = evaluate_graph_preservation(subgraph, final_simplified,name_to_idx)
            all_metrics.append(metrics)
            print("\n=== 图保留质量评估结果 ===")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        except Exception as e:
            print(f"计算评估指标时出错: {e}")
            metrics = {
                'node_preservation': len(core_graph_nodes) / subgraph.vcount(),
                'edge_preservation': final_simplified.ecount() / subgraph.ecount(),
                'laplacian_energy_preservation': current_energy / original_energy_component
            }
            all_metrics.append(metrics)
            print("\n=== 基础评估指标 ===")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    # 合并所有连通分量
    if len(all_simplified_graphs) > 1:
        # 合并所有子图（如果需要全图合并，这里返回所有节点/边的并集）
        combined_graph = all_simplified_graphs[0].copy()
        for g in all_simplified_graphs[1:]:
            combined_graph = combined_graph.disjoint_union(g)
        simplified_graph = combined_graph
        try:
            final_metrics = evaluate_graph_preservation(G_test, combined_graph,name_to_idx) # maybe wrong
        except Exception as e:
            print(f"计算整体评估指标时出错: {e}")
            final_metrics = {
                'node_preservation': combined_graph.vcount() / G_test.vcount(),
                'edge_preservation': combined_graph.ecount() / G_test.ecount(),
                'laplacian_energy_preservation': calculate_laplacian_energy(combined_graph, set(combined_graph.vs["name"]),name_to_idx) / original_energy
            }
    else:
        simplified_graph = all_simplified_graphs[0] if all_simplified_graphs else G_test.copy()
        final_metrics = all_metrics[0] if all_metrics else {
            'node_preservation': 1.0,
            'edge_preservation': 1.0,
            'laplacian_energy_preservation': 1.0
        }

    inference_time = time.time() - start_time
    print(f"\n推理完成，用时: {inference_time:.2f}秒")

    return simplified_graph, final_metrics, inference_time, phase_stats