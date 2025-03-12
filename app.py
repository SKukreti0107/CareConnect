from flask import Flask, request, jsonify
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://medimind-phi.vercel.app/", "http://localhost:3000"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize components
embeddings = download_hugging_face_embeddings()
index_name = "care-connect"

# Load existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Set up retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=1024,
    timeout=120,
    max_retries=3,
    top_p=0.9,
    top_k=40
)

# Set up prompt template
system_prompt = """
You are a knowledgeable medical assistant providing accurate information based on medical documents.
Context: {context}

Please provide a clear, accurate, and well-structured response following these guidelines:
- Focus on medical facts from the provided context
- Use professional yet understandable language
- Include relevant medical terms with brief explanations
- If the information is not in the context, clearly state that
- For conditions/treatments, mention important disclaimers when appropriate

Answer: """

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
            
        question = data['question']
        response = rag_chain.invoke({"input": question})
        
        return jsonify({
            'answer': response['answer']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Update to use environment variables for port and host
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)