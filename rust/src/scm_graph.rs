// ! Fast SCM graph algorithms

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use petgraph::algo::is_cyclic_directed;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Fast topological sort implementation
pub fn topological_sort_fast(
    edges: Vec<(usize, usize)>,
    num_nodes: usize,
) -> PyResult<Vec<usize>> {
    let mut graph = DiGraph::<(), ()>::new();
    
    // Add nodes
    let node_map: HashMap<usize, NodeIndex> = (0..num_nodes)
        .map(|i| (i, graph.add_node(())))
        .collect();
    
    // Add edges
    for (from, to) in edges {
        if let (Some(&from_idx), Some(&to_idx)) = (node_map.get(&from), node_map.get(&to)) {
            graph.add_edge(from_idx, to_idx, ());
        }
    }
    
    // Check for cycles
    if is_cyclic_directed(&graph) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Graph contains cycles - not a valid DAG"
        ));
    }
    
    // Perform topological sort
    let mut topo = Topo::new(&graph);
    let mut result = Vec::new();
    
    while let Some(node_idx) = topo.next(&graph) {
        // Find original node ID
        for (&id, &idx) in &node_map {
            if idx == node_idx {
                result.push(id);
                break;
            }
        }
    }
    
    Ok(result)
}

/// Check d-separation in DAG
pub fn check_d_separation(
    edges: Vec<(usize, usize)>,
    x: Vec<usize>,
    y: Vec<usize>,
    z: Vec<usize>,
) -> bool {
    // Simplified d-separation check
    // In production, would implement full Bayes-Ball algorithm
    
    // For now, simple path checking
    let mut graph = DiGraph::<(), ()>::new();
    let node_map: HashMap<usize, NodeIndex> = (0..100)
        .map(|i| (i, graph.add_node(())))
        .collect();
    
    for (from, to) in edges {
        if let (Some(&from_idx), Some(&to_idx)) = (node_map.get(&from), node_map.get(&to)) {
            graph.add_edge(from_idx, to_idx, ());
        }
    }
    
    // Placeholder - would need full d-separation algorithm
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_sort() {
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let result = topological_sort_fast(edges, 3).unwrap();
        
        // 0 should come before 1 and 2
        assert!(result.iter().position(|&x| x == 0).unwrap() <
                result.iter().position(|&x| x == 1).unwrap());
        assert!(result.iter().position(|&x| x == 1).unwrap() <
                result.iter().position(|&x| x == 2).unwrap());
    }

    #[test]
    fn test_cyclic_detection() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let result = topological_sort_fast(edges, 3);
        assert!(result.is_err());
    }
}
