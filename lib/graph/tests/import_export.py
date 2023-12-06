import os
import pytest
import polars as pl
from graph.graph import Graph, ExportFileFormat

# Sample data for testing
edge_list_data = [["A", "B"], ["B", "C"]]
node_list_data = [["A", 1], ["B", 2], ["C", 3]]

# Create a temporary directory for testing
tmp_path = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(tmp_path, exist_ok=True)


# Test for Edge List Import and Export
def test_edge_list_import_export():
    graph = Graph()

    # Import edge list
    graph.import_edges_from_edge_list(edge_list_data, source_target_col=(0, 1))

    # Export edge list
    exported_data = graph.export_edges_as_edge_list(ExportFileFormat.LIST)

    if not isinstance(exported_data, list):
        raise AssertionError("Exported data is not a list")

    # Verify the data is consistent
    assert exported_data == edge_list_data


# Test for Node List Import and Export
def test_node_list_import_export():
    graph = Graph()

    # Import node list
    graph.import_nodes_from_node_list(node_list_data, node_identifier_col=0)

    # Export node list
    exported_data = graph.export_nodes_as_node_list(ExportFileFormat.LIST)

    if not isinstance(exported_data, list):
        raise AssertionError("Exported data is not a list")

    # Verify the data is consistent
    assert exported_data == node_list_data


def test_csv_import_export_with_headers():
    # Create a temporary CSV file with headers
    csv_file_path = os.path.join(tmp_path, "test_csv_import_export_with_headers_1.csv")
    with open(csv_file_path, "w") as f:
        f.write("source,target\nA,B\nB,C")

    graph = Graph()
    graph.import_edges_from_edge_list(
        str(csv_file_path), source_target_col=("source", "target")
    )

    # Export to a new CSV file
    exported_csv_file = os.path.join(
        tmp_path, "test_csv_import_export_with_headers_2.csv"
    )
    graph.export_edges_as_edge_list(ExportFileFormat.CSV, str(exported_csv_file))

    # Read the exported data
    with open(exported_csv_file, "r") as f:
        exported_data = f.read()

        # Define expected data string
        expected_data = (
            "source,target\nA,B\nB,C\n"  # Ensure this matches the exact format
        )

        # Verify the data
        assert exported_data == expected_data


def test_edges_with_features():
    edge_list_with_features = [["A", "B", 1.0, "type1"], ["B", "C", 0.5, "type2"]]
    graph = Graph()
    graph.import_edges_from_edge_list(
        edge_list_with_features,
        source_target_col=(0, 1),
        edge_attributes=[2, 3],
    )
    exported_data = graph.export_edges_as_edge_list(ExportFileFormat.LIST)
    if not isinstance(exported_data, list):
        raise AssertionError("Exported data is not a list")
    assert exported_data == edge_list_with_features


def test_nodes_with_features():
    node_list_with_features = [
        ["A", 1, "group1"],
        ["B", 2, "group2"],
        ["C", 3, "group3"],
    ]
    graph = Graph()
    graph.import_nodes_from_node_list(
        node_list_with_features,
        node_identifier_col=0,
        features_cols=[1, 2],
    )
    exported_data = graph.export_nodes_as_node_list(ExportFileFormat.LIST)
    if not isinstance(exported_data, list):
        raise AssertionError("Exported data is not a list")
    assert exported_data == node_list_with_features


def test_csv_import_export_without_headers():
    # Create a temporary CSV file with headers
    csv_file_path = os.path.join(
        tmp_path, "test_csv_import_export_without_headers_1.csv"
    )
    with open(csv_file_path, "w") as f:
        f.write("A,B\nB,C")

    graph = Graph()
    graph.import_edges_from_edge_list(str(csv_file_path), source_target_col=(0, 1))

    # Export to a new CSV file
    exported_csv_file = os.path.join(
        tmp_path, "test_csv_import_export_without_headers_2.csv"
    )
    graph.export_edges_as_edge_list(ExportFileFormat.CSV, str(exported_csv_file))

    # Read the exported data
    with open(exported_csv_file, "r") as f:
        exported_data = f.read()

        # Define expected data string
        expected_data = "A,B\nB,C"  # Ensure this matches the exact format

        # Verify the data
        assert exported_data == expected_data or exported_data == expected_data + "\n"


def test_csv_import_export_without_headers_with_features():
    # Create a temporary CSV file with headers
    csv_file_path = os.path.join(
        tmp_path, "test_csv_import_export_without_headers_with_features_1.csv"
    )
    with open(csv_file_path, "w") as f:
        f.write("A,B,1.0,type1\nB,C,0.5,type2")

    graph = Graph()
    graph.import_edges_from_edge_list(
        str(csv_file_path), source_target_col=(0, 1), edge_attributes=[2, 3]
    )

    # Export to a new CSV file
    exported_csv_file = os.path.join(
        tmp_path, "test_csv_import_export_without_headers_with_features_2.csv"
    )
    graph.export_edges_as_edge_list(ExportFileFormat.CSV, str(exported_csv_file))

    # Read the exported data
    with open(exported_csv_file, "r") as f:
        exported_data = f.read()

        # Define expected data string
        expected_data = "A,B,1.0,type1\nB,C,0.5,type2"

        # Verify the data
        assert exported_data == expected_data or exported_data == expected_data + "\n"


def test_csv_import_export_with_headers_with_features():
    # Create a temporary CSV file with headers
    csv_file_path = os.path.join(
        tmp_path, "test_csv_import_export_with_headers_with_features_1.csv"
    )
    with open(csv_file_path, "w") as f:
        f.write("source,target,weight,type\nA,B,1.0,type1\nB,C,0.5,type2")

    graph = Graph()
    graph.import_edges_from_edge_list(
        str(csv_file_path),
        source_target_col=("source", "target"),
        edge_attributes=["weight", "type"],
    )

    # Export to a new CSV file
    exported_csv_file = os.path.join(
        tmp_path, "test_csv_import_export_with_headers_with_features_2.csv"
    )
    graph.export_edges_as_edge_list(ExportFileFormat.CSV, str(exported_csv_file))

    # Read the exported data
    with open(exported_csv_file, "r") as f:
        exported_data = f.read()

        # Define expected data string
        expected_data = "source,target,weight,type\nA,B,1.0,type1\nB,C,0.5,type2"

        # Verify the data
        assert exported_data == expected_data or exported_data == expected_data + "\n"


def test_add_and_get_node():
    graph = Graph()
    graph.nodes = pl.DataFrame({"Node": ["B"], "feature1": [1], "feature2": ["test"]})
    graph.add_node("A", feature1=1, feature2="test")

    node_data = graph.get_node("A")
    assert node_data == {"Node": ["A"], "feature1": [1], "feature2": ["test"]}


def test_delete_node():
    graph = Graph()
    graph.nodes = pl.DataFrame({"Node": ["A", "B"]})
    graph.edges = pl.DataFrame({"Source": ["A"], "Target": ["B"]})

    graph.delete_node("A")

    assert "A" not in graph.nodes["Node"].to_list()
    assert graph.edges.shape[0] == 0  # Edge involving 'A' should be deleted


def test_add_and_get_edge():
    graph = Graph()
    graph.edges = pl.DataFrame(
        {"Source": ["C"], "Target": ["D"], "weight": [0.5]}
    )  # Initialize empty edges DataFrame
    graph.add_edge("A", "B", weight=0.5)

    edge_data = graph.get_edge("A", "B")
    assert edge_data == {"Source": ["A"], "Target": ["B"], "weight": [0.5]}


def test_delete_edge():
    graph = Graph()
    graph.edges = pl.DataFrame({"Source": ["A", "C"], "Target": ["B", "D"]})

    graph.delete_edge("A", "B")

    assert (
        len(
            graph.edges.filter(
                (pl.col("Source") == "A") & (pl.col("Target") == "B")
            ).to_dict(as_series=False)["Source"]
        )
        == 0
    )


def test_get_neighbors():
    graph = Graph()
    graph.edges = pl.DataFrame({"Source": ["A", "A"], "Target": ["B", "C"]})

    neighbors = graph.get_neighbors("A")
    assert set(neighbors) == {"B", "C"}


def test_delete_node_and_edges():
    graph = Graph()
    graph.nodes = pl.DataFrame({"Node": ["A", "B", "C"]})
    graph.edges = pl.DataFrame(
        {"Source": ["A", "B", "C"], "Target": ["B", "C", "A"]}
    )  # Edges form a cycle

    graph.delete_node("A")

    # Check if node 'A' is deleted
    assert "A" not in graph.nodes["Node"].to_list()

    # Check if edges related to 'A' are deleted
    remaining_edges = graph.edges.to_dict(as_series=False)
    assert "A" not in remaining_edges["Source"] and "A" not in remaining_edges["Target"]
