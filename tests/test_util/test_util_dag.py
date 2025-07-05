import pytest
from datavac.util import dag

# Test for include_all_descendants
def test_include_all_descendants_simple():
    graph = {
        'A': ['B', 'C'],
        'B': ['D'],
        'C': [],
        'D': []
    }
    starters = ['A']
    result = dag.include_all_descendants(starters, graph)
    assert result == {'A', 'B', 'C', 'D'}

def test_include_all_descendants_multiple_starters():
    graph = {
        1: [2, 3],
        2: [4],
        3: [],
        4: []
    }
    starters = [1, 3]
    result = dag.include_all_descendants(starters, graph)
    assert result == {1, 2, 3, 4}

def test_reverse_dag_simple():
    graph = {
        'A': ['B', 'C'],
        'B': ['C'],
        'C': []
    }
    reversed_graph = dag.reverse_dag(graph)
    assert reversed_graph == {
        'B': ['A'],
        'C': ['A', 'B'],
        'A': []
    }

def test_reverse_dag_empty():
    graph = {}
    reversed_graph = dag.reverse_dag(graph)
    assert reversed_graph == {}
