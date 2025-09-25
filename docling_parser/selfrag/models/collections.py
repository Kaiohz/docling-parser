from dataclasses import dataclass
from typing import List


@dataclass
class MatchingCollections:
    """
    Represents a collection of matching collection names.

    Attributes:
        matching_collections (List[str]): A list of matching collection names.
    """

    matching_collections: List[str]
