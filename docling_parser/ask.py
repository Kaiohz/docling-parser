from dto.agent.agent import AgentParams
from selfrag.models.self_rag_input import SelfRagInput
from retriever import generate_answer
from agent_factory import AgentFactory
import asyncio
import os

source = "docs/sample.pdf" 
os.environ["GOOGLE_API_KEY"] = "key"


async def main():
    question = "Comment embaucher un auxiliaire de vacances ? Donne moi les details stp"
    # response = await ask_vector_store("Comment embaucher un auxiliaire de vacances ? Donne moi les detals stp")
    # print("---" * 10)
    # print(response)
    # print("---" * 10)
    selfrag_input = SelfRagInput(
        question=question,
        graph_name="AlfredSelfRag",
        embeddings_model="models/text-embedding-004",
        retriever_name="MergerRetriever",
        score_threshold=0.8,
        top_k=10,
    )
    agent_params = AgentParams(agent_name="AgentSelfRag",input_data=selfrag_input.dict(), model_name="gemini-2.0-flash",stream_mode=["updates"])
    agent = AgentFactory(agent_params.agent_name).create_agent(agent_params).get_agent()
    results = await agent.ainvoke(input={"input_data": selfrag_input.dict()}, stream_mode=agent_params.stream_mode) # type: ignore
    docs =  [doc.page_content for doc in results[1][1]['retrieve']['documents']]

    print("---" * 10)
    print(await generate_answer(question, docs))
    print("---" * 10)

if __name__ == "__main__":
    asyncio.run(main())