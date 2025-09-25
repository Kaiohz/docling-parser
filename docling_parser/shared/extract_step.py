def extract_step_from_chunk(chunk_data: dict) -> str | None:
    """
    Extracts the step information from any node in the chunk data.

    Args:
        chunk_data: The dictionary containing node data

    Returns:
        str | None: The step value if found, None otherwise
    """
    if chunk_data[1] and isinstance(chunk_data[1], dict):
        for node_name, node_data in chunk_data[1].items():
            if isinstance(node_data, dict) and "step" in node_data:
                if node_data["step"]:
                    return node_data["step"]
    return None
