from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

connection_string = "postgresql+psycopg://docling:docling@localhost:5432/docling"

async def ask_vector_store(query: str, k: int = 100) -> str:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="connect_rh_adp",
        connection=connection_string,
        async_mode=True,
        use_jsonb=True
    )
    results = await vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k}).ainvoke(query)

    model = GoogleGenerativeAI(model="gemini-2.0-flash")  # Adjust model name as needed

    prompt = (
        "Based on the following documents, provide a concise and accurate answer to the question below.\n"
        "If the documents do not contain relevant information, respond with 'I don't know'.\n\n"
        "Answer in French.\n\n"
        f"Documents: {results}\n"
    )

    model_response = await model.ainvoke(f"{prompt}\nQuestion: {query}\nAnswer:")

    answer = model_response.content if hasattr(model_response, "content") else str(model_response) # type: ignore

    return answer