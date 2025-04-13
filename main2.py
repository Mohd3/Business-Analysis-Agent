import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from typing import Dict, Any
from typing import TypedDict
from typing import Annotated
import os
import pdfplumber
import tempfile

# RAG components
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Import config
from config import DB_CONFIG, API_KEYS

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=API_KEYS["google_api_key"]
)

# Initialize SQL Database
db = SQLDatabase.from_uri(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['name']}",
    sample_rows_in_table_info=3
)

# Initialize TavilySearch
search_tool = TavilySearch(
    max_results=5,
    topic="news",
    tavily_api_key=API_KEYS["tavily_api_key"]
)


# SQL functions
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(question):
    prompt_obj = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": question,
    })
    prompt_text = "\n".join([msg.content for msg in prompt_obj.messages])
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt_text)
    return result["query"]

def execute_query(query):
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return execute_query_tool.invoke(query)

def generate_answer(question, query, result):
    # Convert result to string if it isn't already
    result_str = str(result) if not isinstance(result, str) else result
    
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question in business-friendly language.\n\n"
        f"Question: {question}\n"
        f"SQL Query: {query}\n"
        f"SQL Result: {result_str}"
    )
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Could not generate answer: {str(e)}"

def process_question(question):
    try:
        query = write_query(question)
        result = execute_query(query)
        answer = generate_answer(question, query, result)
        return query, result, answer
    except Exception as e:
        raise Exception(f"Error processing question: {str(e)}")

def sql_ans(input: str) -> Dict[str, Any]:
    try:
        query, result, answer = process_question(input)
        return {
            "answer": answer,
            "query": query,
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}

sql_tool = Tool.from_function(
    name="sql_tool",
    description="EXCLUSIVELY for MZ_Neural Company's internal database queries.",
    func=sql_ans
)

# RAG functions
def load_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def initialize_rag(uploaded_files):
    if not uploaded_files:
        return None
    
    # Save files temporarily
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for file in uploaded_files:
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(path)
    
    # Process files
    documents = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
        os.remove(path)  # Clean up
    os.rmdir(temp_dir)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )
    
    return qa_chain

def determine_tool(question: str) -> str:
    """Determine which tool to use or return 'direct' for simple responses."""
    prompt = f"""Analyze this user input and decide how to respond:
    
User Input: "{question}"

Options:
1. 'sql' - For questions about MZ Neural's internal data (sales, projects, etc.)
2. 'web' - For general/competitor info needing web search
3. 'direct' - For greetings/simple questions that don't need tools

Respond with ONLY one of these words: sql, web, direct"""
    
    response = llm.invoke(prompt)
    tool = response.content.strip().lower() if hasattr(response, 'content') else str(response).strip().lower()
    return tool if tool in ['sql', 'web', 'direct'] else 'web'

# --- Streamlit App ---
st.set_page_config(page_title="AI Business Analyst", layout="wide")
st.title("🧠 Business Analysis Agent")

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Sidebar
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDFs for RAG",
        accept_multiple_files=True,
        type=['pdf']
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                st.session_state.qa_chain = initialize_rag(uploaded_files)
                st.session_state.processed_files = [f.name for f in uploaded_files]
            st.success("Documents processed!")
        else:
            st.warning("Please upload files first")
    
    if st.session_state.processed_files:
        st.subheader("📑 Processed Documents")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")

# Main content
tab1, tab2 = st.tabs(["💬 Query Interface", "ℹ️ System Info"])

with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Enter your question:",
            placeholder="Ask about your data or competitor news...",
            label_visibility="collapsed"
        )
    with col2:
        submit_btn = st.button("Submit", type="primary", use_container_width=True)
    
    # Add RAG checkbox above the results
    use_rag = st.checkbox("Search in uploaded documents", 
                         disabled=not st.session_state.qa_chain,
                         help="Check to search through your uploaded PDFs")

    if submit_btn and user_input:
        with st.spinner("Processing your request..."):
            try:
                # First check if user wants to use RAG
                if use_rag and st.session_state.qa_chain:
                    result = st.session_state.qa_chain.invoke({"query": user_input})
                    st.success("Answer from documents:")
                    st.write(result["result"])
                    
                    with st.expander("Relevant Passages"):
                        retriever = st.session_state.qa_chain.retriever
                        docs = retriever.invoke(user_input)
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Passage {i+1}:**")
                            st.info(doc.page_content)
                            st.divider()
                
                # If not using RAG, use automatic routing
                else:
                    if use_rag and not st.session_state.qa_chain:
                        st.warning("No documents available - using automatic search")
                    
                    selected_tool = determine_tool(user_input)
                    
                    if selected_tool == "direct":
                        response = llm.invoke(f"Respond to this user message appropriately: {user_input}")
                        st.success(response.content if hasattr(response, 'content') else str(response))
                    
                    elif selected_tool == "sql":
                        response = sql_tool.invoke(user_input)
                        if "answer" in response:
                            st.success("Answer:")
                            st.write(response["answer"])
                        else:
                            st.warning("No answer was generated")
                        
                        with st.expander("Technical Details"):
                            if "query" in response:
                                st.write("**SQL Query:**")
                                st.code(response["query"], language="sql")
                            if "result" in response:
                                st.write("**Results:**")
                                st.write(response["result"])
                            if "error" in response:
                                st.error(f"Error: {response['error']}")
                    st.caption(f"🔍 Detected query type: {selected_tool.upper()} response")

                    if selected_tool == "web":
                        search_results = search_tool.invoke(user_input)
                        
                        # Prepare sources for the report
                        sources = [f"{i+1}. [{res['title']}]({res['url']})" 
                                for i, res in enumerate(search_results["results"])]
                        
                        # Create a prompt for the LLM to generate a business report
                        report_prompt = f"""Create a short professional business analysis report based on these competitor news.
                        
                    User Query: {user_input}

                    Search Results:
                    {chr(10).join([f"## {res['title']}{chr(10)}{res.get('content', 'No content available')}" 
                                for res in search_results["results"]])}

                    Instructions:
                    1. Start with "## (suitable header)" 
                    2. Provide a short 2-3 paragraph executive summary
                    3. Include key findings with bullet points
                    4. include strategic insights if and only if applicable (always assume that my company is in the same industry as the competitor am searching about), examples: 
                    - Spoted Strategic Moves? New product launches → Time to analyze: should I respond? Improve? Counter?
                    - Catched a Weaknesses Early (like bad reviews or layoffs)? These are opportunity windows — to steal market share, talent, or customers.
                    - Inspire Ideas, competitor moves might trigger ideas for features or tweaks that we never thought of.

                    5. Use professional but concise business language (don't yap alot)
                    6. Highlight any numbers/statistics found
                    7. Format in clean Markdown"""

                        with st.spinner("Generating professional report..."):
                            try:
                                # Generate the report
                                report = llm.invoke(report_prompt)
                                report_content = report.content if hasattr(report, 'content') else str(report)
                                
                                # Add sources to the report
                                full_report = f"{report_content}\n\n### Sources\n" + "\n".join(sources)
                                
                                # Display the report
                                st.markdown(full_report, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Failed to generate report: {str(e)}")
                                # Fallback to simple display
                                for result in search_results["results"]:
                                    st.markdown(f"### [{result['title']}]({result['url']})")
                                    st.write(result.get('content', 'No content available'))
                                    st.divider()
                
                    st.caption(f"🔍 Detected query type: {selected_tool.upper()} response")
            
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                st.exception(e)    

with tab2:
    st.subheader("System Configuration")
    st.json({
        "LLM": "gemini-2.0-flash",
        "Embeddings": "all-mpnet-base-v2",
        "Vector Store": "FAISS",
        "Database": f"MySQL ({db_name})",
        "Web Search": "Tavily",
        "Routing": "Automatic tool selection"
    })