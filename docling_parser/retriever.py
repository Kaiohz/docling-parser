from langchain_postgres.vectorstores import PGVector
from llm.chat_client_factory import ChatClientFactory
from llm.embeddings_client_factory import EmbeddingsClientFactory

connection_string = "postgresql+psycopg://docling:docling@localhost:5432/docling"

async def ask_vector_store(query: str, k: int = 100) -> str:
    embeddings = EmbeddingsClientFactory(provider="Ollama", model="nomic-embed-text:v1.5").create_client()

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="connect_rh_adp",
        connection=connection_string,
        async_mode=True,
        use_jsonb=True
    )
    results = await vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k}).ainvoke(query)

    model = ChatClientFactory(provider="Ollama", temperature=0.0, model="qwen2.5:3b").create_client()

    prompt = (
        "Based on the following documents, provide a detailed, accurate, and comprehensive answer to the question below.\n"
        "Include examples, thorough explanations, and reference the documents when possible.\n"
        "If the documents do not contain relevant information, respond with 'I don't know'.\n\n"
        "Answer in French.\n\n"
        f"Documents: {results}\n"
    )

    model_response = await model.ainvoke(f"{prompt}\nQuestion: {query}\nAnswer:")

    answer = model_response.content if hasattr(model_response, "content") else str(model_response) # type: ignore

    return answer # type: ignore

async def generate_answer(question,results, model) -> str:
    results = [result.page_content for result in results]
    prompt = (
        "Based on the following documents, provide an accurate, and comprehensive answer to the question below.\n"
        "Include examples, thorough explanations, and reference the documents when possible.\n"
        "Just provide the answer without adding any additional commentary.\n"
        "If the documents do not contain relevant information, respond that you don't know'.\n\n"
        "Answer in French.\n\n"
        f"Question: {question}\n\n"
        f"Context: {results}\n"
    )
    model = ChatClientFactory(provider="Google", temperature=0.0, model=model).create_client()
    model_response = await model.ainvoke(prompt)
    answer = model_response.content if hasattr(model_response, "content") else str(model_response) # type: ignore
    return answer # type: ignore