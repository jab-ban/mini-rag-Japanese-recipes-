from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()  # تحميل القيم من .env
api_key = os.getenv("GROQ_API_KEY")

os.environ['GROQ_API_KEY'] = api_key

import os
api_key = os.getenv("GROQ_API_KEY")

# API Key

# Initialize LLM
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.4,
    max_tokens=2000
)

# Connect to existing Chroma DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vdb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
retriever = vdb.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Function to get recipe answer
def get_recipe(user_input):
    # Retrieve top 3 similar documents
    results = retriever.get_relevant_documents(user_input)
    docs_text = "\n".join([doc.page_content for doc in results])

    # Build prompt for LLM
    prompt = f"You are a chef assistant. User asked: {user_input}\n\nRelevant recipes from DB:\n{docs_text}\n\nAnswer with the best recipe."
    
    # Generate response
    response = llm.invoke(prompt)
    return response.content

# Interactive test
print("Chef-Sensei in your service 🍳👨‍🍳 :")
print("Ask anything about recipes. Type 'exit' to quit.\n")

while True:
    user_input = input("Your question: ").strip()
    if user_input.lower() == "exit":
        print("Jaa ne! 👋")
        break
    if user_input:
        answer = get_recipe(user_input)
        print(f"\nAnswer: {answer}\n")

        

