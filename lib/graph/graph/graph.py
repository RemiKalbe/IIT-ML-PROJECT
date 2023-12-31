import polars as pl
import pandas as pd
import os
from typing import List, Union, Dict, Any, NamedTuple, Tuple, TypeVar, Generic
from enum import Enum, auto
import numpy as np


class ExportFileFormat(Enum):
    CSV = auto()
    JSON = auto()
    POLARS_DF = auto()
    PANDAS_DF = auto()
    LIST = auto()
    NUMPY = auto()


class NodeMetadata(NamedTuple):
    node_identifier_col: Union[str, int, None]
    features_cols: Union[List[Union[str, int]], None]
    class_col: Union[str, int, None]
    delimiter: str
    has_header: bool


class EdgeMetadata(NamedTuple):
    source_target_col: Union[Tuple[str, str], Tuple[int, int], None]
    edge_attributes: Union[List[Union[str, int]], None]
    delimiter: str
    has_header: bool


class Graph:
    """
    Graph class to represent a graph using Polars DataFrames.

    Attributes:
        nodes (pl.DataFrame): DataFrame to store nodes and their attributes.
        edges (pl.DataFrame): DataFrame to store edges and their attributes.
    """

    def __init__(self) -> None:
        """
        Initializes the Graph class.
        """
        self.nodes: Union[pl.DataFrame, None] = None
        self.edges: Union[pl.DataFrame, None] = None
        self.node_metadata: Union[NodeMetadata, None] = None
        self.edge_metadata: Union[EdgeMetadata, None] = None

    def import_edges_from_edge_list(
        self,
        data: Union[
            str, List[List[Any]], np.ndarray[Any, Any], pd.DataFrame, pl.DataFrame
        ],
        source_target_col: Union[Union[Tuple[str, str], Tuple[int, int]], None] = None,
        override_data_file_extension: Union[str, None] = None,
        delimiter: str = ",",
        edge_attributes: Union[List[Union[str, int]], None] = None,
        infer_nodes: bool = False,
    ) -> None:
        """
        Imports graph data from an edge list.

        Args:
            - data: The edge list data or path to the edge list file.
                    It can be a file path (csv, txt, parquet) or a data structure (List, Numpy Array, Pandas DataFrame, Polars DataFrame) containing the edge list.
            - source_target_col: (Optional) Tuple specifying the source and target columns.
                              If strings are provided, a header is assumed. If integers are provided, no header is assumed.
            - override_data_file_extension: (Optional) Override the file extension for file paths. This is useful when the file extension does not match the file format.
            - delimiter: (Optional) The delimiter used in CSV files.
            - edge_attributes: (Optional) List of additional attribute names or indexes for edges.
            - infer_nodes: (Optional) Indicates whether to infer nodes from the edge list.

        """
        # Security check, if source_target_col is provided and it's a tuple of strings,
        # and edge_attributes is provided, and it's a tuple of integers, or vice versa,
        # raise an error
        if source_target_col and edge_attributes:
            if all(isinstance(col, str) for col in source_target_col) and all(
                isinstance(col, int) for col in edge_attributes
            ):
                raise ValueError(
                    "source_target_col is a tuple of strings, but edge_attributes is a tuple of integers"
                )
            elif all(isinstance(col, int) for col in source_target_col) and all(
                isinstance(col, str) for col in edge_attributes
            ):
                raise ValueError(
                    "source_target_col is a tuple of integers, but edge_attributes is a tuple of strings"
                )

        has_header = (
            isinstance(source_target_col[0], str) if source_target_col else True
        )

        if isinstance(data, str):  # File path is provided
            _, file_extension = (
                os.path.splitext(data)
                if override_data_file_extension is None
                else (None, override_data_file_extension)
            )

            if file_extension in [".csv", ".txt"]:
                data = pl.read_csv(data, separator=delimiter, has_header=has_header)
            elif file_extension == ".parquet":
                data = pl.read_parquet(data)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

        elif isinstance(data, (pd.DataFrame, np.ndarray, list)):
            data = pl.DataFrame(data)  # Convert to Polars DataFrame

        if isinstance(data, pl.DataFrame):
            # Standardize column names based on source_target_col or assume first two columns
            # Check if source_target_col is provided
            if source_target_col:
                # Handle based on whether column identifiers are strings or integers
                if all(isinstance(col, str) for col in source_target_col):
                    # String names for columns
                    # Ensure that source_target_col is a tuple of strings
                    if not isinstance(source_target_col, tuple):
                        raise TypeError("source_target_col must be a tuple")
                    if not all(isinstance(col, str) for col in source_target_col):
                        raise TypeError("source_target_col must be a tuple of strings")

                    source_col_name, target_col_name = source_target_col
                    source_col_name, target_col_name = str(source_col_name), str(
                        target_col_name
                    )  # Pylance can't infer that the types are strings

                    data = data.rename(
                        {source_col_name: "Source", target_col_name: "Target"}
                    )
                elif all(isinstance(col, int) for col in source_target_col):
                    # Integer indices for columns
                    # Ensure that source_target_col is a tuple of integers
                    if not isinstance(source_target_col, tuple):
                        raise TypeError("source_target_col must be a tuple")
                    if not all(isinstance(col, int) for col in source_target_col):
                        raise TypeError("source_target_col must be a tuple of integers")

                    source_idx, target_idx = source_target_col
                    source_idx, target_idx = int(source_idx), int(
                        target_idx
                    )  # Pylance can't infer that the types are integers

                    columns = list(data.columns)
                    source_col_name, target_col_name = (
                        columns[source_idx],
                        columns[target_idx],
                    )
                    data = data.rename(
                        {source_col_name: "Source", target_col_name: "Target"}
                    )
            else:
                # Default to first two columns as source and target
                columns = list(data.columns)
                source_col_name, target_col_name = columns[0], columns[1]
                data = data.rename(
                    {source_col_name: "Source", target_col_name: "Target"}
                )

            # Handling edge attributes
            if edge_attributes:
                if all(isinstance(col, int) for col in edge_attributes):
                    # Integer indices for columns
                    # Rename to "Feature i" where i is the index
                    fn = 0
                    for i, col in enumerate(data.columns):
                        # If the current column is not in the edge_attributes list, drop it
                        if col not in ["Source", "Target"] and i not in edge_attributes:
                            data.drop_in_place(col)
                        elif col not in ["Source", "Target"]:
                            data = data.rename({col: f"Feature {fn}"})
                            fn += 1
                elif all(isinstance(col, str) for col in edge_attributes):
                    # Delete columns that are not in the edge_attributes list
                    for col in data.columns:
                        if (
                            col not in ["Source", "Target"]
                            and col not in edge_attributes
                        ):
                            data.drop_in_place(col)

            self.edges = data

            # Optionally infer nodes from the edge list
            if infer_nodes:
                # Extract unique node identifiers from the Source and Target columns
                unique_nodes = pl.concat(
                    [
                        data.select("Source").rename({"Source": "Node"}),
                        data.select("Target").rename({"Target": "Node"}),
                    ]
                ).unique()

                # Update or create the nodes DataFrame
                if not isinstance(self.nodes, pl.DataFrame):
                    self.nodes = unique_nodes
                else:
                    # Merge with existing nodes DataFrame to ensure no duplicates
                    self.nodes = self.nodes.vstack(unique_nodes).unique()

            # Store the metadata
            self.edge_metadata = EdgeMetadata(
                source_target_col=source_target_col,
                edge_attributes=edge_attributes,
                delimiter=delimiter,
                has_header=has_header,
            )

    def import_nodes_from_node_list(
        self,
        data: Union[
            str, List[List[Any]], np.ndarray[Any, Any], pd.DataFrame, pl.DataFrame
        ],
        node_identifier_col: Union[str, int, None] = None,
        features_cols: Union[List[Union[str, int]], None] = None,
        class_col: Union[str, int, None] = None,
        override_data_file_extension: Union[str, None] = None,
        delimiter: str = ",",
    ) -> None:
        """
        Imports node data from a node list.

        Args:
            data: The node list data or path to the node list file.
                  It can be a file path (csv, txt, parquet) or a data structure (List, Numpy Array, Pandas DataFrame, Polars DataFrame) containing the node list.
            node_identifier_col: (Optional) The column name or index for the node identifiers.
            features_cols: (Optional) List of additional attribute names or indexes for node features.
            override_data_file_extension: (Optional) Override the file extension for file paths.
            delimiter: (Optional) The delimiter used in CSV files.
        """
        # Security check, making sure that node_identifier_col,
        # features_cols, and class_col have all one type of column identifier
        # raise an error
        is_node_identifier_col_str = (
            isinstance(node_identifier_col, str) if node_identifier_col else None
        )
        is_features_cols_str = (
            all(isinstance(col, str) for col in features_cols)
            if features_cols
            else None
        )
        is_class_col_str = isinstance(class_col, str) if class_col else None

        all_cols = [is_node_identifier_col_str, is_features_cols_str, is_class_col_str]

        count_not_none = sum(1 for x in all_cols if x is not None)

        if count_not_none > 0:
            # Then the sum of either True or False is 0. If it's 0, then all the values are the same type
            t_s = sum(1 for x in all_cols if x is True)
            f_s = sum(1 for x in all_cols if x is False)
            if t_s != 0 and f_s != 0:
                raise ValueError(
                    "node_identifier_col, features_cols, and class_col must all be of the same type"
                )

        has_header = (
            isinstance(node_identifier_col, str)
            if node_identifier_col is not None
            else True
        )

        if isinstance(data, str):  # File path is provided
            _, file_extension = (
                os.path.splitext(data)
                if override_data_file_extension is None
                else (None, override_data_file_extension)
            )

            if file_extension in [".csv", ".txt"]:
                data = pl.read_csv(data, separator=delimiter, has_header=has_header)
            elif file_extension == ".parquet":
                data = pl.read_parquet(data)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

        elif isinstance(data, (pd.DataFrame, np.ndarray, list)):
            data = pl.DataFrame(data)  # Convert to Polars DataFrame

        if isinstance(data, pl.DataFrame):
            # Standardize column names
            if node_identifier_col is not None:
                if isinstance(node_identifier_col, int):
                    # Integer index for column
                    node_col_name = data.columns[node_identifier_col]
                    node_col_idx = node_identifier_col
                    data = data.rename({node_col_name: "Node"})
                else:
                    # String name for column
                    node_col_idx = data.columns.index(node_identifier_col)
                    data = data.rename({node_identifier_col: "Node"})
            else:
                # Default to first column as node identifier
                node_col_name = data.columns[0]
                node_col_idx = 0
                data = data.rename({node_col_name: "Node"})

            # Handling node features
            if features_cols:
                if all(isinstance(col, int) for col in features_cols):
                    # Integer indices for columns
                    # Rename to "Feature i" where i is the index
                    fn = 0
                    for i, col in enumerate(data.columns):
                        # If the current column is not in the features_cols list, drop it
                        if i in features_cols:
                            data = data.rename({col: f"Feature {fn}"})
                            fn += 1
                elif all(isinstance(col, str) for col in features_cols):
                    # Delete columns that are not in the features_cols list
                    for col in data.columns:
                        if col != "Node" and col not in features_cols:
                            data.drop_in_place(col)
            else:
                # Determine indices for the features
                # taking into account the node identifier column
                # and the class column
                indices = [i for i in range(len(data.columns))]
                if node_identifier_col is not None:
                    indices.remove(node_col_idx)
                if class_col is not None:
                    if isinstance(class_col, int):
                        indices.remove(class_col)
                    else:
                        indices.remove(data.columns.index(class_col))

                # Rename to "Feature i" where i is the index
                fn = 0
                for i, col in enumerate(data.columns):
                    # If the current column is not in the indices list, drop it
                    if i in indices:
                        data = data.rename({col: f"Feature {fn}"})
                        fn += 1

            # Handling class column
            if class_col:
                if isinstance(class_col, int):
                    # Integer index for column
                    class_col_name = data.columns[class_col]
                    data = data.rename({class_col_name: "Class"})
                else:
                    # String name for column
                    data = data.rename({class_col: "Class"})

            self.nodes = data

            # Store the metadata
            self.node_metadata = NodeMetadata(
                node_identifier_col=node_identifier_col,
                features_cols=features_cols,
                class_col=class_col,
                delimiter=delimiter,
                has_header=has_header,
            )

    def export_edges_as_edge_list(
        self, file_format: ExportFileFormat, file_path: Union[str, None] = None
    ) -> Union[None, List, np.ndarray, pd.DataFrame, pl.DataFrame]:
        """
        Exports the edge data as an edge list, supporting various file formats.

        Args:
            file_format: The format to export the edge data.
            file_path: (Optional) The file path to save the edge data. If None, returns the data in the specified format.

        Returns:
            The edge data in the specified format, or None if written to a file.
        """
        if not isinstance(self.edges, pl.DataFrame):
            raise ValueError("No edge data is available")
        if not isinstance(self.edge_metadata, EdgeMetadata):
            raise ValueError("Edge metadata is not defined")

        edge_data = self.edges.clone()

        source_idx = edge_data.find_idx_by_name("Source")
        target_idx = edge_data.find_idx_by_name("Target")
        feature_idx = [
            edge_data.find_idx_by_name(f"Feature {i}")
            for i in range(self.get_edge_feature_count())
        ]

        if self.edge_metadata.has_header:
            assert isinstance(self.edge_metadata.source_target_col, tuple)

            (
                original_source_col,
                original_target_col,
            ) = self.edge_metadata.source_target_col
            original_feature_cols = self.edge_metadata.edge_attributes

            assert original_source_col is not None
            assert original_target_col is not None
            assert original_feature_cols is not None

            # If has_header is True, then source_target_col and edge_attributes are strings
            assert isinstance(original_source_col, str)
            assert isinstance(original_target_col, str)
            assert all(isinstance(col, str) for col in original_feature_cols)

            for c_idx in edge_data.columns:
                if c_idx == source_idx:
                    edge_data = edge_data.rename(mapping={c_idx: original_source_col})
                elif c_idx == target_idx:
                    edge_data = edge_data.rename(mapping={c_idx: original_target_col})
                elif c_idx in feature_idx:
                    c = original_feature_cols.pop(0)
                    assert isinstance(c, str)
                    edge_data = edge_data.rename(mapping={c_idx: c})

        return self._export_data_as_list(
            data=edge_data,
            metadata=self.edge_metadata,
            file_format=file_format,
            file_path=file_path,
        )

    def export_nodes_as_node_list(
        self, file_format: ExportFileFormat, file_path: Union[str, None] = None
    ) -> Union[None, List, np.ndarray, pd.DataFrame, pl.DataFrame]:
        """
        Exports the node data as a node list, supporting various file formats.

        Args:
            file_format: The format to export the node data.
            file_path: (Optional) The file path to save the node data. If None, returns the data in the specified format.

        Returns:
            The node data in the specified format, or None if written to a file.
        """
        if not isinstance(self.nodes, pl.DataFrame):
            raise ValueError("Nodes data frame is not defined.")
        if not isinstance(self.node_metadata, NodeMetadata):
            raise ValueError("Node metadata is not defined.")

        node_data = self.nodes.clone()

        # Identifying the index of the node identifier column
        node_idx = node_data.find_idx_by_name("Node")
        feature_idx = [
            node_data.find_idx_by_name(f"Feature {i}")
            for i in range(self.get_node_feature_count())
        ]
        class_idx = node_data.find_idx_by_name("Class")

        if self.node_metadata.has_header:
            original_node_identifier_col = self.node_metadata.node_identifier_col
            original_feature_cols = self.node_metadata.features_cols
            original_class_col = self.node_metadata.class_col

            assert original_class_col is not None
            assert original_feature_cols is not None
            assert original_node_identifier_col is not None

            # If has_header is True, then node_identifier_col, features_cols, and class_col are strings
            assert isinstance(original_node_identifier_col, str)
            assert all(isinstance(col, str) for col in original_feature_cols)
            assert isinstance(original_class_col, str)

            for c_idx in node_data.columns:
                if c_idx == node_idx:
                    node_data = node_data.rename(
                        mapping={c_idx: original_node_identifier_col}
                    )
                elif c_idx in feature_idx:
                    c = original_feature_cols.pop(0)
                    assert isinstance(c, str)
                    node_data = node_data.rename(mapping={c_idx: c})
                elif c_idx == class_idx:
                    node_data = node_data.rename(mapping={c_idx: original_class_col})

        return self._export_data_as_list(
            data=node_data,
            metadata=self.node_metadata,
            file_format=file_format,
            file_path=file_path,
        )

    def _export_data_as_list(
        self,
        data: pl.DataFrame,
        metadata: Union[NodeMetadata, EdgeMetadata],
        file_format: ExportFileFormat,
        file_path: Union[str, None],
    ) -> Union[None, List, np.ndarray, pd.DataFrame, pl.DataFrame]:
        """
        Helper function to export data as a list, considering the specified metadata and format.

        Args:
            data: The DataFrame to export.
            metadata: Metadata containing information about the original format.
            file_format: The specified format for export.
            file_path: The destination file path for the data.

        Returns:
            The data in the specified format, or None if written to a file.
        """
        if file_format == ExportFileFormat.CSV and file_path:
            data.write_csv(
                file_path,
                separator=metadata.delimiter,
                include_header=metadata.has_header,
            )
        elif file_format == ExportFileFormat.JSON and file_path:
            data.write_json(file_path)
        elif file_format == ExportFileFormat.POLARS_DF:
            return data
        elif file_format == ExportFileFormat.PANDAS_DF:
            return data.to_pandas()
        elif file_format == ExportFileFormat.LIST:
            return data.to_numpy().tolist()
        elif file_format == ExportFileFormat.NUMPY:
            return data.to_numpy()
        else:
            raise ValueError("Unsupported output format")

    def add_node(
        self, node: Union[str, int], **attributes: Union[str, int, float]
    ) -> None:
        """
        Adds a node to the graph with optional additional attributes.

        Args:
            node: Unique identifier for the node.
            **attributes: Arbitrary number of keyword arguments for node attributes.
        """
        new_node = {"Node": node, **attributes}
        if not isinstance(self.nodes, pl.DataFrame):
            raise ValueError("Add node called without existing nodes DataFrame")
        self.nodes = self.nodes.vstack(pl.DataFrame([new_node]))

    def delete_node(self, node: Union[str, int]) -> None:
        """
        Deletes a node from the graph. Also deletes any edges that contain the node.

        Args:
            node: The node identifier to remove.
        """
        if not isinstance(self.nodes, pl.DataFrame):
            raise ValueError("Remove node called without existing nodes DataFrame")
        if not isinstance(self.edges, pl.DataFrame):
            raise ValueError("Remove node called without existing edges DataFrame")

        self.nodes = self.nodes.filter(pl.col("Node") != node)
        # Delete edges that contain the node
        self.edges = self.edges.filter(
            (pl.col("Source") != node) & (pl.col("Target") != node)
        )

    def add_edge(
        self,
        source: Union[str, int],
        target: Union[str, int],
        **attributes: Union[str, int, float],
    ) -> None:
        """
        Adds an edge to the graph with optional additional attributes.

        Args:
            source: The source node identifier of the edge.
            target: The target node identifier of the edge.
            **attributes: Arbitrary number of keyword arguments for edge attributes.
        """
        new_edge = {"Source": source, "Target": target, **attributes}
        if not isinstance(self.edges, pl.DataFrame):
            raise ValueError("Add edge called without existing edges DataFrame")
        self.edges = self.edges.vstack(pl.DataFrame([new_edge]))

    def delete_edge(self, source: Union[str, int], target: Union[str, int]) -> None:
        """
        Deletes an edge from the graph.

        Args:
            source: The source node identifier of the edge.
            target: The target node identifier of the edge.
        """
        if not isinstance(self.edges, pl.DataFrame):
            raise ValueError("Remove edge called without existing edges DataFrame")
        self.edges = self.edges.filter(
            (pl.col("Source") != source) & (pl.col("Target") != target)
        )

    def get_neighbors(self, node: Union[str, int]) -> List[Union[str, int]]:
        """
        Retrieves the neighbors of a given node.

        Args:
            node: The node identifier to get neighbors for.

        Returns:
            A list of neighbors for the specified node.
        """
        if not isinstance(self.edges, pl.DataFrame):
            raise ValueError("Get neighbors called without existing edges DataFrame")
        neighbors_as_target = self.edges.filter(pl.col("Source") == node)["Target"]
        neighbors_as_source = self.edges.filter(pl.col("Target") == node)["Source"]
        neighbors = neighbors_as_target.append(neighbors_as_source).unique()
        return neighbors.to_list()

    def get_node(self, node: Union[str, int]) -> Dict[str, Any]:
        """
        Retrieves a node and its attributes.

        Args:
            node: The node identifier.

        Returns:
            A dictionary containing the node and its attributes.
        """
        if not isinstance(self.nodes, pl.DataFrame):
            raise ValueError("Get node called without existing nodes DataFrame")
        attributes = self.nodes.filter(pl.col("Node") == node)
        return attributes.to_dict(as_series=False)

    def get_node_feature_count(self) -> int:
        """
        Retrieves the number of node features.

        Returns:
            The number of node features.
        """
        if not isinstance(self.nodes, pl.DataFrame):
            raise ValueError(
                "Get node feature count called without existing nodes DataFrame"
            )
        c = 0
        for col in self.nodes.columns:
            if col.startswith("Feature"):
                c += 1
        return c

    def get_node_with_most_neighbors(self) -> Union[str, int]:
        """Finds the node with the highest number of neighbors.

        Returns:
            The identifier of the node with the highest number of neighbors.
        """
        if not isinstance(self.edges, pl.DataFrame):
            raise ValueError("Edges DataFrame is not defined.")

        # Getting counts of each node as a source and as a target in the edges.
        target_counts = (
            self.edges.groupby("Target")
            .count()
            .rename({"Target": "Node", "count": "target_count"})
        )
        source_counts = (
            self.edges.groupby("Source")
            .count()
            .rename({"Source": "Node", "count": "source_count"})
        )

        # Merging the counts together into a single DataFrame
        all_counts_df = target_counts.join(
            source_counts, on="Node", how="outer"
        ).with_columns(
            (pl.col("target_count") + pl.col("source_count")).alias("total_count")
        )

        # Identifying the node with the maximum count (degree)
        max_degree_row = (
            all_counts_df.sort("total_count", descending=True).limit(1)["Node"].item()
        )

        return max_degree_row

    def get_edge_two_ways(
        self, node_1: Union[str, int], node_2: Union[str, int]
    ) -> Union[Dict[str, Any], None]:
        """
        Retrieves an edge and its attributes, what ever the direction of the edge is.

        Args:
            node_1: The node identifier of the edge.
            node_2: The node identifier of the edge.

        Returns:
            A dictionary containing the edge and its attributes.
        """
        if not isinstance(self.edges, pl.DataFrame):
            raise ValueError("Get edge called without existing edges DataFrame")
        print(f"node_1: {node_1}")
        print(f"node_2: {node_2}")
        edge_1 = self.get_edge(node_1, node_2)
        edge_2 = self.get_edge(node_2, node_1)
        print(f"edge_1: {edge_1}")
        print(f"edge_2: {edge_2}")

        # Remove either edge_1 or edge_2 if they are the same but in different directions
        if edge_1 and edge_2 and edge_1 == edge_2:
            edge_2 = None
        return edge_1 or edge_2

    def get_edge(
        self, source: Union[str, int], target: Union[str, int]
    ) -> Union[Dict[str, Any], None]:
        """
        Retrieves an edge and its attributes.

        Args:
            source: The source node identifier of the edge.
            target: The target node identifier of the edge.

        Returns:
            A dictionary containing the edge and its attributes.
        """
        if not isinstance(self.edges, pl.DataFrame):
            raise ValueError("Get edge called without existing edges DataFrame")
        edge = self.edges.filter(
            (pl.col("Source") == source) & (pl.col("Target") == target)
        ).limit(1)

        return edge.to_dict(as_series=False) if not edge.is_empty() else None

    def get_edge_feature_count(self) -> int:
        """
        Retrieves the number of edge features.

        Returns:
            The number of edge features.
        """
        if not isinstance(self.edges, pl.DataFrame):
            raise ValueError(
                "Get edge feature count called without existing edges DataFrame"
            )
        c = 0
        for col in self.edges.columns:
            if col.startswith("Feature"):
                c += 1
        return c
