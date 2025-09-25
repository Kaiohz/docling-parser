import json
from response_strategy import ResponseStrategy


class GenericResponseStrategy(ResponseStrategy):
    """
    Concrete strategy for processing generic agent responses.
    """

    async def execute(self, chunk) -> str: # type: ignore
        """
        Processes a generic answer stream and returns a formatted chunk as a JSON string.

        Args:
            chunk (Any): The chunk of data to process.

        Returns:
            str | None: The formatted response as a JSON string, or None if not applicable.
        """
        if isinstance(chunk, dict):
            result = json.dumps(chunk)
            return result
        return chunk 
