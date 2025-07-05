from typing import Hashable, Mapping, Sequence, TypeVar


T=TypeVar('T',bound=Hashable)
def include_all_descendants(starters: list[T], graph: Mapping[T,Sequence[T]] ):
    all_required = set()

    def dfs(node):
        all_required.add(node)
        for des in graph.get(node, []):
            if des not in all_required:
                dfs(des)

    for start_node in starters: dfs(start_node)
    return all_required

T2=TypeVar('T2',bound=Hashable)
def reverse_dag(graph: Mapping[T2,Sequence[T]]) -> Mapping[T,Sequence[T2]]:
    """Reverses the direction of the edges in a directed acyclic graph (DAG).

    Note: in the type annotation, it is assumed that isinstance(T,T2) is True,
    ie any value in the original graph is a valid key in the reversed graph.
    
    Args:
        graph: A mapping where keys are nodes and values are sequences of nodes that the key points to.
    
    Returns:
        A new mapping where the direction of edges is reversed.
    """
    reversed_graph: Mapping[T,list[T2]] = {}
    for node, successors in graph.items():
        if node not in reversed_graph:
            reversed_graph[node] = [] # type: ignore # see note about T and T2
        for successor in successors:
            if successor not in reversed_graph:
                reversed_graph[successor] = []
            reversed_graph[successor].append(node)
    return reversed_graph