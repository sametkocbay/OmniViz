import pandas as pd
import re


def parse_fortran_float(s):
    """
    Parse a float string that may have malformed scientific notation.
    Handles cases like '1.39615174-309' which should be '1.39615174E-309'
    """
    s = s.strip()
    if not s:
        return float('nan')

    # Check for malformed scientific notation: number followed by +/- without E
    # Pattern: digits with decimal, then +/- followed by digits (missing E)
    pattern = r'^([+-]?\d+\.?\d*)([-+])(\d+)$'
    match = re.match(pattern, s)
    if match:
        mantissa, sign, exp = match.groups()
        s = f"{mantissa}E{sign}{exp}"

    try:
        return float(s)
    except ValueError:
        return float('nan')


def read_patran_neutral(filename):
    """
    Robustly reads a Patran Neutral file (.msh/.out)
    """
    nodes_list = []  # List to store [node_id, x, y, z]
    elements_hex = []  # To store 8-node hexahedra

    # Helper to safely parse ID strings that might look like "1.000"
    def parse_id(val):
        return int(float(val))

    with open(filename, "r") as f:
        # Use an iterator to handle large files without loading all into RAM
        f_iter = iter(f)

        try:
            for line in f_iter:
                line = line.strip()
                if not line: continue

                parts = line.split()

                # Read label
                try:
                    label = parse_id(parts[0])
                except ValueError:
                    continue

                # --- Packet 25: Title ---
                if label == 25:
                    next(f_iter)  # Skip Title Line
                    continue

                # --- Packet 26: Summary ---
                if label == 26:
                    # Skip Summary Data line + Date line
                    # The current line was the label 26 + counts, so we just skip the Date
                    next(f_iter)
                    continue

                # --- Packet 1: Node Data ---
                if label == 1:
                    # Line 1: 1 <NodeID>
                    node_id = parse_id(parts[1])

                    # Line 2: x y z
                    xyz_line = next(f_iter).strip()
                    coords = list(map(float, xyz_line.split()))

                    # Line 3: Constraints (Metadata) - Skip
                    next(f_iter)

                    nodes_list.append([node_id] + coords)
                    continue

                # --- Packet 2: Element Data ---
                if label == 2:
                    # Line 1: 2 <ElemID> <ShapeID> <Nodes> ...
                    # elem_id = parse_id(parts[1])
                    shape_id = parse_id(parts[2])

                    # Line 2: Config data (Metadata) - Skip
                    next(f_iter)

                    # Line 3: Connectivity
                    conn_line = next(f_iter).strip()
                    node_ids = list(map(parse_id, conn_line.split()))

                    # Hex
                    if shape_id == 8:
                        elements_hex.append(node_ids[:8])

                    continue

                # Recognize end of file
                if label == 99:
                    break

        except StopIteration:
            pass  # End of file reached unexpectedly

    # Create DataFrame
    df_nodes = pd.DataFrame(nodes_list, columns=["node_id", "x", "y", "z"])

    # Ensure indices are integers
    df_nodes["node_id"] = df_nodes["node_id"].astype(int)
    df_nodes = df_nodes.sort_values("node_id").reset_index(drop=True)

    return df_nodes, elements_hex


def read_xyz_data(filename):
    """
    Reads a simple XYZ coordinate file where each line contains space-separated x, y, z values.
    Handles malformed scientific notation (e.g., '1.39615174-309' instead of '1.39615174E-309')

    Args:
        filename: Path to the file containing XYZ data

    Returns:
        pd.DataFrame: DataFrame with columns ['x', 'y', 'z']
    """
    xyz_list = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split and convert to float using robust parser
            coords = line.split()
            if len(coords) >= 3:
                x = parse_fortran_float(coords[0])
                y = parse_fortran_float(coords[1])
                z = parse_fortran_float(coords[2])
                xyz_list.append([x, y, z])

    df_xyz = pd.DataFrame(xyz_list, columns=["x", "y", "z"])

    # Filter out invalid values (nan, inf)
    df_xyz = df_xyz.replace([float('inf'), float('-inf')], float('nan'))
    df_xyz = df_xyz.dropna()

    return df_xyz


def read_vector_field(filename):
    """
    Reads a 6-column vector field file with columns: x y z Bx By Bz.
    Lines starting with '#' are treated as comments and skipped.
    Handles malformed scientific notation (e.g. '1.234-309' instead of '1.234E-309').

    Args:
        filename: Path to the file

    Returns:
        pd.DataFrame: DataFrame with columns ['x', 'y', 'z', 'Bx', 'By', 'Bz']
    """
    rows = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            values = [parse_fortran_float(p) for p in parts[:6]]
            rows.append(values)

    df = pd.DataFrame(rows, columns=["x", "y", "z", "Bx", "By", "Bz"])
    df = df.replace([float('inf'), float('-inf')], float('nan'))
    df = df.dropna()

    return df


def detect_file_columns(filename):
    """
    Returns the number of data columns in a file by inspecting the first
    non-blank, non-comment line.

    Args:
        filename: Path to file

    Returns:
        int: Number of columns detected, or 0 if the file is unreadable.
    """
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                return len(line.split())
    except Exception:
        pass
    return 0
