from retriever import ask_vector_store
import asyncio
import os

source = "docs/sample.pdf" 
os.environ["GOOGLE_API_KEY"] = "key"


async def main():
    response = await ask_vector_store("Je n'arrive pas a me connecter a connect rh")
    print("Response from vector store:", response)
    # toc_map now contains the desired structure

if __name__ == "__main__":
    asyncio.run(main())