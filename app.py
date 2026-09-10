from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv 
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import embedding
from src.prompt import sys_prompt


app = Flask(__name__)

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

if os.getenv("HF_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

docs = Chroma(persist_directory="vectordb", embedding_function=embedding)
retriever = docs.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_prompt),
        ("human", "{input}")
    ]
)

QA_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, QA_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        if request.is_json:
            data = request.get_json()
            user_msg = data.get("msg", "")
        else:
            user_msg = request.form.get("msg", "")

        if not user_msg:
            return "Empty message", 400

        print(f"User Input: {user_msg}")
        
        response = rag_chain.invoke({"input": user_msg})
        answer = response.get("answer", "No answer generated.")
        
        print(f"Response: {answer}")
        return str(answer)

    except Exception as e:
        print(f"Error occurred: {e}")
        return f"Server Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)