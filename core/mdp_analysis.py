"""
MDP Analysis Module for Robotic Surgery Plan Certification

Based on the framework from ifacconf-v2.tex:
- Feasibility Analysis: absorption stability + finite-time reachability
- Robustness Analysis: ρ(i,j) index for each transition
- Recovery: Find attractors in invariant subsets and suggest recovery edges
"""

from collections import deque
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import sys

# Increase recursion limit for Tarjan's algorithm on large graphs
sys.setrecursionlimit(10000)


def build_adjacency_from_transitions(transitions: np.ndarray, include_all_states: bool = False) -> Dict[int, List[int]]:
    """
    Convert transition count matrix to adjacency list.
    
    Args:
        transitions: N x N matrix where transitions[i][j] = count of i->j transitions
        include_all_states: If True, include all states even if they have no transitions.
                           If False, only include states that have at least one transition.
    
    Returns:
        Adjacency list {node: [successors]}
    """
    n = len(transitions)
    graph = {}
    
    # First pass: find states with outgoing transitions
    for i in range(n):
        successors = []
        for j in range(n):
            if transitions[i][j] > 0:
                successors.append(j)
        if successors or include_all_states:
            graph[i] = successors
    
    # Second pass: ensure all destination states are in the graph
    # (states that are reached but have no outgoing edges, like target)
    all_destinations = set()
    for successors in graph.values():
        all_destinations.update(successors)
    
    for dest in all_destinations:
        if dest not in graph:
            graph[dest] = []
    
    return graph


def compute_transition_probability_matrix(transitions: np.ndarray) -> np.ndarray:
    """
    Convert transition counts to probability matrix P_π.
    
    P_π(s_i, s_j) = count(i→j) / sum_k(count(i→k))
    """
    n = len(transitions)
    prob_matrix = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        row_sum = np.sum(transitions[i])
        if row_sum > 0:
            prob_matrix[i] = transitions[i] / row_sum
    
    return prob_matrix


def build_reverse_graph(graph: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """Build reverse graph where edges point from children to parents."""
    reverse = {}
    for node in graph:
        if node not in reverse:
            reverse[node] = []
        for child in graph[node]:
            if child not in reverse:
                reverse[child] = []
    for node, children in graph.items():
        for child in children:
            reverse[child].append(node)
    return reverse


def get_all_nodes(graph: Dict[int, List[int]]) -> Set[int]:
    """Get all nodes in the graph."""
    nodes = set(graph.keys())
    for children in graph.values():
        nodes.update(children)
    return nodes


def find_nodes_reaching_target(graph: Dict[int, List[int]], target: int) -> Set[int]:
    """Find all nodes that can reach the target using BFS on reverse graph."""
    reverse_graph = build_reverse_graph(graph)
    reachable = set()
    queue = deque([target])
    reachable.add(target)
    
    while queue:
        node = queue.popleft()
        for parent in reverse_graph.get(node, []):
            if parent not in reachable:
                reachable.add(parent)
                queue.append(parent)
    
    return reachable


def compute_reachability_layers(graph: Dict[int, List[int]], target: int) -> Dict[int, int]:
    """
    Compute reachability layers L[t] as defined in the paper.
    
    L[0] = {s_T}
    L[t] = {s_i | exists s_j in L[t-1] s.t. P_π(s_i, s_j) > 0}
    
    Returns:
        Dictionary mapping each node to its layer (distance to target).
        Nodes that cannot reach target have layer = -1.
    """
    reverse_graph = build_reverse_graph(graph)
    all_nodes = get_all_nodes(graph)
    
    layers = {node: -1 for node in all_nodes}
    layers[target] = 0
    
    queue = deque([target])
    
    while queue:
        node = queue.popleft()
        current_layer = layers[node]
        for parent in reverse_graph.get(node, []):
            if layers[parent] == -1:
                layers[parent] = current_layer + 1
                queue.append(parent)
    
    return layers


def check_feasibility(graph: Dict[int, List[int]], target: int) -> Tuple[bool, Dict]:
    """
    Check feasibility conditions from Theorem 1:
    1. Absorption: P_π(s_T, s_T) = 1 (target has no outgoing edges to other states)
    2. Finite-time reachability: all states can reach s_T within T_max steps
    
    Returns:
        (is_feasible, details_dict)
    """
    all_nodes = get_all_nodes(graph)
    n = len(all_nodes)
    
    # Condition 1: Absorption - target has no outgoing edges (or only self-loop)
    target_in_graph = target in graph
    target_successors = graph.get(target, [])
    has_absorption = len(target_successors) == 0 or (len(target_successors) == 1 and target_successors[0] == target)
    
    # Debug output
    print(f"[check_feasibility] Target state: {target}")
    print(f"[check_feasibility] Target in graph: {target_in_graph}")
    print(f"[check_feasibility] Target successors: {target_successors}")
    print(f"[check_feasibility] Has absorption: {has_absorption}")
    
    # Condition 2: Finite-time reachability
    reachable = find_nodes_reaching_target(graph, target)
    unreachable = all_nodes - reachable
    has_reachability = len(unreachable) == 0
    
    print(f"[check_feasibility] All nodes count: {len(all_nodes)}")
    print(f"[check_feasibility] Reachable count: {len(reachable)}")
    print(f"[check_feasibility] Unreachable count: {len(unreachable)}")
    if unreachable:
        print(f"[check_feasibility] Unreachable nodes (first 10): {sorted(list(unreachable))[:10]}")
    
    # Compute layers for T_max estimation
    layers = compute_reachability_layers(graph, target)
    max_layer = max(l for l in layers.values() if l >= 0) if any(l >= 0 for l in layers.values()) else 0
    
    is_feasible = has_absorption and has_reachability
    
    return is_feasible, {
        'absorption': has_absorption,
        'reachability': has_reachability,
        'unreachable_states': sorted(list(unreachable)),
        'max_layer': max_layer,
        'layers': layers,
        'node_count': n,
    }


def find_invariant_subset(graph: Dict[int, List[int]], target: int, edge: Tuple[int, int]) -> Set[int]:
    """
    Find invariant subset Ω(i,j) after removing edge (i→j).
    
    Ω(i,j) = states that cannot reach target after edge removal
    
    IMPORTANT: This only counts nodes that become NEWLY unreachable after edge removal.
    If a node was already unreachable before edge removal, it's not part of the invariant subset.
    """
    source, dest = edge
    
    # Find nodes reachable BEFORE edge removal
    reachable_before = find_nodes_reaching_target(graph, target)
    
    # Create modified graph with edge removed
    modified_graph = {}
    for node, children in graph.items():
        if node == source:
            modified_graph[node] = [c for c in children if c != dest]
        else:
            modified_graph[node] = children.copy()
    
    # Ensure all nodes present
    all_nodes = get_all_nodes(graph)
    for node in all_nodes:
        if node not in modified_graph:
            modified_graph[node] = []
    
    # Find nodes that can still reach target AFTER edge removal
    reachable_after = find_nodes_reaching_target(modified_graph, target)
    
    # Invariant subset = nodes that WERE reachable but NOW cannot reach target
    # (not nodes that were already unreachable)
    return reachable_before - reachable_after


def find_unreachable_subsets(graph: Dict[int, List[int]], target: int) -> List[List[int]]:
    """
    Find all connected components of nodes that cannot reach the target.
    
    Returns:
        List of connected components (each is a list of node IDs)
    """
    all_nodes = get_all_nodes(graph)
    reachable = find_nodes_reaching_target(graph, target)
    unreachable = all_nodes - reachable
    
    if not unreachable:
        return []
    
    # Find connected components within unreachable nodes (undirected connectivity)
    visited = set()
    components = []
    
    for start_node in unreachable:
        if start_node in visited:
            continue
        
        # BFS to find connected component
        component = set()
        queue = deque([start_node])
        component.add(start_node)
        
        while queue:
            node = queue.popleft()
            # Check outgoing edges
            for neighbor in graph.get(node, []):
                if neighbor in unreachable and neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)
            # Check incoming edges (reverse graph)
            for other_node, children in graph.items():
                if other_node in unreachable and node in children and other_node not in component:
                    component.add(other_node)
                    queue.append(other_node)
        
        visited.update(component)
        components.append(sorted(list(component)))
    
    return components


def find_scc_tarjan(graph: Dict[int, List[int]], nodes_subset: Set[int]) -> List[List[int]]:
    """
    Find strongly connected components using Tarjan's algorithm.
    Only considers nodes in nodes_subset.
    
    Returns:
        List of SCCs (each is a list of node IDs)
    """
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    sccs = []
    
    def strongconnect(node):
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True
        
        for neighbor in graph.get(node, []):
            if neighbor not in nodes_subset:
                continue
            if neighbor not in index:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif on_stack.get(neighbor, False):
                lowlinks[node] = min(lowlinks[node], index[neighbor])
        
        if lowlinks[node] == index[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == node:
                    break
            # Include all SCCs (even single nodes, they're attractors)
            sccs.append(sorted(scc))
    
    for node in nodes_subset:
        if node not in index:
            strongconnect(node)
    
    return sccs


def find_attractors_in_subset(
    graph: Dict[int, List[int]], 
    subset: List[int], 
    all_reachable: Set[int]
) -> List[List[int]]:
    """
    Find attractor nodes in an unreachable subset.
    An attractor is a node/SCC that has no outgoing edges to outside the SCC.
    
    Returns:
        List of attractors (each is a list of node IDs in the SCC)
    """
    subset_set = set(subset)
    
    # Find SCCs in the subset
    sccs = find_scc_tarjan(graph, subset_set)
    
    # Find attractors (SCCs with no outgoing edges to other nodes outside the SCC)
    attractors = []
    for scc in sccs:
        scc_set = set(scc)
        has_external_edge = False
        for node in scc:
            for neighbor in graph.get(node, []):
                if neighbor not in scc_set:
                    has_external_edge = True
                    break
            if has_external_edge:
                break
        
        if not has_external_edge:
            # This SCC is an attractor (bottom SCC)
            attractors.append(scc)
    
    # If no true attractors found, take nodes with no outgoing edges
    if not attractors:
        for node in subset:
            outgoing = [n for n in graph.get(node, []) if n in subset_set]
            if not outgoing:
                attractors.append([node])
    
    # If still no attractors, just use all nodes
    if not attractors:
        attractors = [[n] for n in subset]
    
    return attractors


def state_to_grid(state_id: int, ny: int) -> Tuple[int, int]:
    """Convert state ID to grid coordinates (i, j)."""
    i = state_id // ny
    j = state_id % ny
    return (i, j)


def grid_distance(state1: int, state2: int, ny: int) -> float:
    """Compute Euclidean distance between two states in grid space."""
    i1, j1 = state_to_grid(state1, ny)
    i2, j2 = state_to_grid(state2, ny)
    return ((i1 - i2) ** 2 + (j1 - j2) ** 2) ** 0.5


def find_nearest_reachable_state(
    source_node: int, 
    reachable: Set[int],
    target: int,
    ny: int
) -> Optional[int]:
    """
    Find the nearest reachable state to connect from source_node.
    Uses physical grid distance (Euclidean) to find the closest valid state.
    
    Args:
        source_node: The unreachable node that needs a recovery edge
        reachable: Set of nodes that can reach the target
        target: The target state
        ny: Number of columns in the grid (for coordinate conversion)
    
    Returns:
        Node ID of the physically nearest reachable state, or None if none available
    """
    if not reachable:
        return None
    
    # Exclude target itself for better graph structure (prefer connecting to intermediate states)
    candidates = [n for n in reachable if n != target]
    
    if not candidates:
        # If only target is reachable, use target
        return target
    
    # Find the physically closest reachable state
    min_dist = float('inf')
    nearest = None
    
    for candidate in candidates:
        dist = grid_distance(source_node, candidate, ny)
        if dist < min_dist:
            min_dist = dist
            nearest = candidate
    
    return nearest


def compute_recovery_plan(graph: Dict[int, List[int]], target: int, ny: int = 10) -> Dict:
    """
    Compute minimal recovery plan for an infeasible network.
    
    For each unreachable subset:
    1. Find attractors (SCCs with no outgoing edges)
    2. Suggest adding an edge from a node in the attractor to the PHYSICALLY nearest reachable node
    
    Args:
        graph: Adjacency list of the transition graph
        target: The target/absorbing state
        ny: Number of columns in the grid (for computing physical distances)
    
    Returns:
        Dict with unreachable_subsets, recovery_edges, and statistics
    """
    all_nodes = get_all_nodes(graph)
    reachable = find_nodes_reaching_target(graph, target)
    unreachable_subsets = find_unreachable_subsets(graph, target)
    layers = compute_reachability_layers(graph, target)
    
    recovery_edges = []
    
    for subset in unreachable_subsets:
        # Find attractors in this subset
        attractors = find_attractors_in_subset(graph, subset, reachable)
        
        for attractor in attractors:
            # Pick a node from the attractor
            source_node = attractor[0]
            
            # Find physically nearest reachable node (using grid distance)
            dest_node = find_nearest_reachable_state(source_node, reachable, target, ny)
            
            if dest_node is not None:
                # Compute physical distance for info
                dist = grid_distance(source_node, dest_node, ny)
                recovery_edges.append({
                    'source': source_node,
                    'dest': dest_node,
                    'dest_layer': layers.get(dest_node, -1),
                    'physical_distance': round(dist, 2),
                    'subset': subset,
                    'attractor': attractor,
                })
    
    return {
        'unreachable_subsets': unreachable_subsets,
        'recovery_edges': recovery_edges,
        'total_unreachable': sum(len(s) for s in unreachable_subsets),
        'edges_needed': len(recovery_edges),
    }


def apply_recovery_edges(
    graph: Dict[int, List[int]], 
    recovery_edges: List[Dict]
) -> Dict[int, List[int]]:
    """
    Apply recovery edges to the graph and return a new (recovered) graph.
    """
    recovered_graph = {k: v.copy() for k, v in graph.items()}
    
    for edge in recovery_edges:
        source = edge['source']
        dest = edge['dest']
        if source not in recovered_graph:
            recovered_graph[source] = []
        if dest not in recovered_graph[source]:
            recovered_graph[source].append(dest)
    
    return recovered_graph


def compute_robustness_index(graph: Dict[int, List[int]], target: int, edge: Tuple[int, int]) -> Tuple[float, Set[int]]:
    """
    Compute robustness index ρ(i,j) for a transition.
    
    ρ(i,j) = 1 - |Ω(i,j)| / (|S| - 1)
    
    Returns:
        (robustness_index, invariant_subset)
    """
    invariant = find_invariant_subset(graph, target, edge)
    n = len(get_all_nodes(graph))
    
    if n <= 1:
        return 1.0, invariant
    
    rho = 1.0 - len(invariant) / (n - 1)
    return rho, invariant


def analyze_all_transitions(graph: Dict[int, List[int]], target: int) -> List[Dict]:
    """
    Analyze robustness for all transitions in the graph.
    
    Returns:
        List of dicts with edge, robustness, invariant_subset info
    """
    results = []
    n = len(get_all_nodes(graph))
    
    # Debug: Count nodes with multiple outgoing edges
    multi_out_nodes = sum(1 for s, succ in graph.items() if len(succ) > 1)
    total_edges = sum(len(succ) for succ in graph.values())
    print(f"[analyze_all_transitions] Total nodes: {n}, Total edges: {total_edges}")
    print(f"[analyze_all_transitions] Nodes with >1 outgoing edge: {multi_out_nodes}")
    
    critical_count = 0
    for source, successors in graph.items():
        for dest in successors:
            edge = (source, dest)
            rho, invariant = compute_robustness_index(graph, target, edge)
            
            is_critical = len(invariant) > 0
            if is_critical:
                critical_count += 1
            
            results.append({
                'edge': list(edge),
                'from_state': source,
                'to_state': dest,
                'robustness': rho,
                'invariant_subset': sorted(list(invariant)),
                'invariant_size': len(invariant),
                'is_critical': is_critical,
                'source_out_degree': len(successors),  # How many outgoing edges the source has
            })
    
    print(f"[analyze_all_transitions] Critical edges: {critical_count}/{total_edges}")
    
    # Sort by robustness (lowest first = most critical)
    results.sort(key=lambda x: (x['robustness'], x['from_state'], x['to_state']))
    
    return results


def full_mdp_analysis(transitions: np.ndarray, target_state: int, ny: int = 10) -> Dict:
    """
    Complete MDP analysis pipeline.
    
    Args:
        transitions: N x N transition count matrix
        target_state: Target/absorbing state index
        ny: Number of columns in the grid (for computing physical distances in recovery)
    
    Returns:
        Complete analysis results dict
    """
    # Build adjacency graph
    graph = build_adjacency_from_transitions(transitions)
    
    # Compute probability matrix
    prob_matrix = compute_transition_probability_matrix(transitions)
    
    # Check feasibility
    is_feasible, feasibility_details = check_feasibility(graph, target_state)
    
    # Analyze all transitions for robustness
    transition_analysis = analyze_all_transitions(graph, target_state)
    
    # Compute layers
    layers = compute_reachability_layers(graph, target_state)
    layers_by_level = {}
    for node, layer in layers.items():
        if layer not in layers_by_level:
            layers_by_level[layer] = []
        layers_by_level[layer].append(node)
    
    # Count statistics
    critical_count = sum(1 for t in transition_analysis if t['is_critical'])
    safe_count = len(transition_analysis) - critical_count
    
    result = {
        'feasible': is_feasible,
        'feasibility': feasibility_details,
        'transition_probability_matrix': prob_matrix.tolist(),
        'transition_analysis': transition_analysis,
        'layers': {int(k): sorted(v) for k, v in layers_by_level.items()},
        'statistics': {
            'total_transitions': len(transition_analysis),
            'critical_transitions': critical_count,
            'safe_transitions': safe_count,
            'max_layer': feasibility_details['max_layer'],
        }
    }
    
    # If not feasible, compute recovery plan with physical distance
    if not is_feasible:
        recovery_plan = compute_recovery_plan(graph, target_state, ny=ny)
        result['recovery'] = {
            'unreachable_subsets': recovery_plan['unreachable_subsets'],
            'recovery_edges': [
                {
                    'source': e['source'],
                    'dest': e['dest'],
                    'dest_layer': e['dest_layer'],
                    'physical_distance': e.get('physical_distance', 0),
                    'attractor': e['attractor'],
                }
                for e in recovery_plan['recovery_edges']
            ],
            'total_unreachable': recovery_plan['total_unreachable'],
            'edges_needed': recovery_plan['edges_needed'],
        }
    
    return result

