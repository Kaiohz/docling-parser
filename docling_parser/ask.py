from retriever import ask_vector_store
import asyncio
import os

source = "docs/sample.pdf" 
os.environ["GOOGLE_API_KEY"] = "key"


async def main():
    response = await ask_vector_store("Comment embaucher un auxiliaire de vacances ? Donne moi les detals stp")
    print("---" * 10)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())