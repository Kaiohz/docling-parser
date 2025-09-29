from dto.agent.agent import AgentParams
from selfrag.models.self_rag_input import SelfRagInput
from retriever import generate_answer
from agent_factory import AgentFactory
import asyncio
import os

os.environ["GOOGLE_API_KEY"] = "key"


async def main(question:str):
    selfrag_input = SelfRagInput(
        question=question,
        graph_name="AlfredSelfRag",
        embeddings_model="models/text-embedding-004",
        retriever_name="MergerRetriever",
        score_threshold=0.8,
        top_k=10,
    )
    agent_params = AgentParams(agent_name="AgentSelfRag",input_data=selfrag_input.dict(), model_name="gemini-2.5-flash-lite",stream_mode=["updates"])
    agent = AgentFactory(agent_params.agent_name).create_agent(agent_params).get_agent()
    results = await agent.ainvoke(input={"input_data": selfrag_input.dict()}, stream_mode=agent_params.stream_mode) # type: ignore
    docs =  [doc for doc in results[1][1]['grade_documents']['documents']] # type: ignore
    metadatas = []
    if docs:
        metadatas = [doc.metadata for doc in docs] # type: ignore
    answer = await generate_answer(question, docs, agent_params.model_name) # type: ignore
    return (answer, metadatas)

if __name__ == "__main__":
        asyncio.run(main(""))