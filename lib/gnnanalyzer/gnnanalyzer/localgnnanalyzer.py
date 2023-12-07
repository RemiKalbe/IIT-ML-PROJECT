from typing import Any, Dict, List, Set, Union

from lib.graph.graph.graph import Graph
from lib.gnnanalyzer.gnnanalyzer.localimpactcalculator import LocalImpactCalculator


class LocalGNNAnalyzer:
    def __init__(
        self, graph: Graph, local_impact_calculator: LocalImpactCalculator
    ) -> None:
        self.graph: Graph = graph
        self.ablation_plan: List[List[List[Union[str, int]]]] = []
        self.ablated_edges: Union[List[Dict[str, Any]], None] = None
        self.leaderboard: List[List[Dict[str, Any]]] = []
        self.current_step: int = 0
        self.current_depth: int = 1
        self.max_depth: int = 1
        self.original_gnn_prediction: Union[Union[float, List[float], str], None] = None
        self.impact_calculator: LocalImpactCalculator = local_impact_calculator
        self.current_starting_node: Union[str, int, None] = None
        self.previous_starting_nodes: Set[Union[str, int]] = set()

    def prepare_ablation_plan(self, starting_node: Union[str, int], max_depth: int = 1):
        self.max_depth = max_depth
        self.ablation_plan = self._generate_ablation_plan(
            starting_node=starting_node, previous_nodes=self.previous_starting_nodes
        )

    def _generate_ablation_plan(
        self,
        starting_node: Union[str, int],
        previous_nodes: Set[Union[str, int]] = set(),
    ) -> List[List[List[Union[str, int]]]]:
        self.current_starting_node = starting_node
        self.previous_starting_nodes.add(starting_node)

        neighbors: List[Union[str, int]] = self.graph.get_neighbors(starting_node)
        plan: List[List[List[Union[str, int]]]] = []

        # Filter out neighbors that are in the previous_nodes set
        neighbors = [n for n in neighbors if n not in previous_nodes]

        for neighbor in neighbors:
            edge = [starting_node, neighbor]
            plan.append([edge])

        return plan

    def execute_ablation_step(
        self, prev_gnn_prediction: Union[float, List[float], str]
    ):
        # Save the original GNN prediction
        if self.original_gnn_prediction is None:
            is_original_prediction = True
            self.original_gnn_prediction = prev_gnn_prediction
        else:
            is_original_prediction = False

        # Re-add the previously ablated edges with their features
        if self.ablated_edges:
            for edge in self.ablated_edges:
                # Extract attributes and re-add the edge
                edge_attributes = {
                    k: v[0] for k, v in edge.items() if k not in ["Source", "Target"]
                }
                self.graph.add_edge(
                    edge["Source"][0], edge["Target"][0], **edge_attributes
                )

        # Perform the current step's ablation
        if self.current_step < len(self.ablation_plan):
            planed_ablation_edges = self.ablation_plan[self.current_step]
            expected_ablated_edges = [
                self.graph.get_edge_two_ways(edge[0], edge[1])
                for edge in planed_ablation_edges
            ]

            if any([edge is None for edge in expected_ablated_edges]):
                raise ValueError(
                    "One or more edges in the ablation plan were not found in the graph."
                )

            self.ablated_edges = expected_ablated_edges  # type: ignore

            for edge in self.ablated_edges:
                self.graph.delete_edge(edge["Source"], edge["Target"])  # type: ignore

            # Calculate impact and update the leaderboard

            if not is_original_prediction:
                impact: float = self.impact_calculator.calculate_impact(
                    prev_prediction=self.original_gnn_prediction,
                    current_prediction=prev_gnn_prediction,
                )
                if len(self.leaderboard) < self.current_depth:
                    self.leaderboard.append([])
                self.leaderboard[self.current_depth - 1].append(
                    {"edges": self.ablated_edges, "impact": impact}
                )

            self.current_step += 1

        # Check if it's time to move to the next depth
        if (
            self.current_step >= len(self.ablation_plan)
            and self.current_depth < self.max_depth
        ):
            self.current_depth += 1
            self.current_step = 0
            next_starting_node = self._determine_next_starting_node(
                is_original_prediction
            )
            self.prepare_ablation_plan(
                next_starting_node, self.max_depth - self.current_depth
            )

    def _determine_next_starting_node(self, is_original_prediction: bool = False):
        """
        Determines the next starting node for ablation based on the leaderboard.

        Returns:
            str: The node identifier of the next starting node.
        """
        # Find the ablated edge with the maximum impact
        max_impact = float("-inf")
        next_edge = None

        for depth_impacts in self.leaderboard:
            print(depth_impacts)
            for impact_info in depth_impacts:
                if impact_info["impact"] > max_impact:
                    max_impact = impact_info["impact"]
                    next_edge = impact_info["edges"]

        # If no edge was found, raise an error
        if not next_edge and not is_original_prediction:
            raise ValueError(
                "No edge was found with the maximum impact. This should never happen."
            )

        # If no edge was found, this could be because this is the original prediction
        # and no edge was ablated. In this case, we can just return the current starting node.
        if not next_edge and self.original_gnn_prediction:
            return self.current_starting_node

        # Choose one of the nodes of the most impactful edge as the next starting node
        if self.current_starting_node is None:
            raise ValueError(
                "The current starting node is None. This should never happen."
            )

        source_node, target_node = next_edge[0]["Source"][0], next_edge[0]["Target"][0]
        return target_node if source_node == self.current_starting_node else source_node

    def has_next_step(self):
        return (
            self.current_step < len(self.ablation_plan)
            and self.current_depth <= self.max_depth
        )

    def get_interpretation(self) -> List[Dict[str, Any]]:
        """
        Retrieves an ordered interpretation of the most influential paths.

        Returns:
            A list of dictionaries, where each dictionary represents an influential path
            with its overall impact score and the individual nodes and edges involved in the path.
        """
        interpretation = []
        # Traverse each depth level in the leaderboard
        print(self.leaderboard)
        for depth, impacts_at_depth in enumerate(self.leaderboard):
            for impact_info in impacts_at_depth:
                # Extract and accumulate the path information
                path = {"impact": impact_info["impact"], "nodes": [], "edges": []}
                for edge_data in impact_info["edges"]:
                    # Edge source and target
                    source_node = edge_data["Source"][0]
                    target_node = edge_data["Target"][0]
                    # Accumulate edges
                    path["edges"].append((source_node, target_node))
                    # Accumulate nodes if not already in the list
                    if source_node not in path["nodes"]:
                        path["nodes"].append(source_node)
                    if target_node not in path["nodes"]:
                        path["nodes"].append(target_node)
                # Append path information to interpretation
                interpretation.append(path)

        # Order interpretation by accumulated impact score
        ordered_interpretation = sorted(
            interpretation, key=lambda x: x["impact"], reverse=True
        )
        return ordered_interpretation
