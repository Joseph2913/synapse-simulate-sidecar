from models.seed_graph import SeedGraph, SimulationNode, SimulationEdge


class InternalGraph:
    """Internal graph representation for the simulation pipeline."""
    def __init__(self):
        self.nodes: dict[str, SimulationNode] = {}
        self.edges: list[SimulationEdge] = []
        self.adjacency: dict[str, list[str]] = {}    # node_id → [connected node_ids]
        self.edge_index: dict[str, list[SimulationEdge]] = {}  # node_id → [edges]


def import_graph(seed_graph: SeedGraph) -> InternalGraph:
    """
    Convert Synapse's SeedGraph into an InternalGraph.

    Key transformations:
    - Build adjacency map for fast traversal
    - Index edges by node for persona relationship summaries
    - No filtering — all nodes included (exclusions handled by Synapse before export)
    """
    graph = InternalGraph()

    for node in seed_graph.nodes:
        graph.nodes[node.id] = node
        graph.adjacency[node.id] = []
        graph.edge_index[node.id] = []

    for edge in seed_graph.edges:
        graph.edges.append(edge)
        # Build adjacency (bidirectional)
        if edge.source_node_id in graph.adjacency:
            graph.adjacency[edge.source_node_id].append(edge.target_node_id)
        if edge.target_node_id in graph.adjacency:
            graph.adjacency[edge.target_node_id].append(edge.source_node_id)
        # Index edges by node
        if edge.source_node_id in graph.edge_index:
            graph.edge_index[edge.source_node_id].append(edge)
        if edge.target_node_id in graph.edge_index:
            graph.edge_index[edge.target_node_id].append(edge)

    return graph
