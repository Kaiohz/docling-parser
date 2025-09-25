from typing import AsyncIterable, Union
from dto.agent.agent_response import AgentResponse
from response_strategy import ResponseStrategy
from selfrag.response import SelfRagResponseStrategy
import asyncio


answer_strategy: dict[str, ResponseStrategy] = {
    "AgentSelfRag": SelfRagResponseStrategy(),
}


def get_strategy(agent_name: str) -> ResponseStrategy:
    """
    Retrieves the response strategy instance for the given agent name.

    Args:
        agent_name (str): The name of the agent.

    Returns:
        ResponseStrategy: The corresponding response strategy instance.
    """
    return answer_strategy[agent_name]


async def generate_answer(stream, agent_name: str) -> Union[AgentResponse, AsyncIterable[str]]:
    """
    Asynchronously generates answers for a given agent by processing a stream of chunks.

    Args:
        stream: An asynchronous iterable of chunk data.
        agent_name (str): The name of the agent to use.

    Yields:
        str: JSON string of the AgentResponse or string for each processed chunk.
    """
    strategy = get_strategy(agent_name)
    async for chunk in stream:
        response: Union[AgentResponse, str] = await strategy.execute(chunk)
        if response and isinstance(response, AgentResponse):
            yield f"{response.model_dump_json()}\n"
        else:
            yield f"{response}\n"
    await asyncio.sleep(0.5)
