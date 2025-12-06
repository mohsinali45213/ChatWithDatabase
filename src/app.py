import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate             
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI            
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pymysql
pymysql.install_as_MySQLdb()

load_dotenv()

def init_database(host, port, user, password, database):
    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_url)


def get_sql_chain(db):
    template = '''
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;

    Your turn:

    Question: {question}
    SQL Query:
    '''

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")     

    def get_schema(_):
        return db.get_table_info()
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}

    NOTE : along with the answer provide table format if it is possible.
    Remember to follow these guidelines:
    - If the SQL query returns no results, respond with "I am sorry, I could not find an answer to your question."
    if there is not table or data to show just say "No data available to display.".
    upderstand that the user might not specify the table name in their question, so use the schema to infer the correct table to query.

    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")      

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars['query'])
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "chat_history": chat_history,
        "question": user_query
    })


st.set_page_config("Chat with MySql Database", page_icon='💬', layout="wide")

# Futuristic Dark Pink Theme CSS
st.markdown("""
<style>
    /* Main color scheme - Dark Pink Palette */
    :root {
        --primary-pink: #ff1493;
        --dark-pink: #c71585;
        --neon-pink: #ff69b4;
        --deep-purple: #1a0033;
        --dark-bg: #0d0015;
        --card-bg: #1a0a2e;
    }
    
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0d0015 0%, #1a0a2e 50%, #2d1b4e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0a2e 0%, #0d0015 100%);
        border-right: 2px solid #ff1493;
        box-shadow: 5px 0 20px rgba(255, 20, 147, 0.3);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #ff69b4;
        text-shadow: 0 0 10px #ff1493;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(90deg, #ff1493, #ff69b4, #ff1493);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem !important;
        text-align: center;
        font-weight: 800;
        letter-spacing: 3px;
        text-shadow: 0 0 30px rgba(255, 20, 147, 0.5);
        animation: glow 2s ease-in-out infinite;
        font-family: 'Courier New', monospace;
    }
    
    @keyframes glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(26, 10, 46, 0.8) !important;
        border: 2px solid #ff1493 !important;
        color: #ff69b4 !important;
        border-radius: 10px;
        padding: 10px;
        font-family: 'Courier New', monospace;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 20px rgba(255, 20, 147, 0.6);
        border-color: #ff69b4 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ff1493, #c71585) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(255, 20, 147, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 20, 147, 0.6);
        background: linear-gradient(135deg, #ff69b4, #ff1493) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(26, 10, 46, 0.6) !important;
        border: 1px solid #ff1493;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(255, 20, 147, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* AI message specific */
    [data-testid="stChatMessageContent"] {
        color: #ff69b4;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Chat input */
    .stChatInput > div > div {
        background: rgba(26, 10, 46, 0.8) !important;
        border: 2px solid #ff1493 !important;
        border-radius: 25px;
        box-shadow: 0 5px 20px rgba(255, 20, 147, 0.3);
    }
    
    .stChatInput input {
        color: #ff69b4 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(0, 255, 136, 0.1) !important;
        border: 1px solid #00ff88 !important;
        color: #00ff88 !important;
        border-radius: 10px;
    }
    
    .stError {
        background: rgba(255, 20, 147, 0.2) !important;
        border: 1px solid #ff1493 !important;
        color: #ff69b4 !important;
        border-radius: 10px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #ff1493 !important;
    }
    
    /* Markdown text */
    p, span, div {
        color: #e6b3d1 !important;
    }
    
    /* Horizontal line */
    hr {
        border-color: #ff1493 !important;
        opacity: 0.5;
    }
    
    /* Labels */
    label {
        color: #ff69b4 !important;
        font-weight: 600;
        font-family: 'Courier New', monospace;
        letter-spacing: 1px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0d0015;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #ff1493, #c71585);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #ff69b4, #ff1493);
    }
    
    /* Decorative elements */
    .decoration {
        position: fixed;
        border-radius: 50%;
        opacity: 0.1;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
</style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [
        AIMessage(content="🌟 Hello! I'm your AI Database Assistant. Ask me anything about your data or structure.")
    ]

st.title("🔮 Chat with MySql Database")


with st.sidebar:
    st.header("🔌 Database Connection")
    st.markdown("---")
    st.markdown("### 📊 Connect to MySQL")
    st.markdown("")

    host = st.text_input("🌐 Host", value="sql8.freesqldatabase.com", key='host')
    port = st.text_input("🔢 Port", value="3306", key='port')
    user = st.text_input("👤 User", value="sql8809948", key='user')
    password = st.text_input("🔐 Password", type="password", value="vFh7Mqw9PP", key='password')
    database = st.text_input("💾 Database", value="sql8809948", key='database')
    
    st.markdown("")

    if st.button("⚡ Connect to Database"):
        with st.spinner("🔄 Establishing connection..."):
            try:
                db = init_database(
                    st.session_state['host'],
                    st.session_state['port'],
                    st.session_state['user'],
                    st.session_state['password'],
                    st.session_state['database']
                )
                st.success("✅ Connected successfully!")
                st.session_state['db'] = db
            except Exception as e:
                st.error(f"❌ Connection error: {e}")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Enter your database credentials
    - Click Connect to establish connection
    - Start chatting with your data!
    """)



# Chat history display with enhanced styling
for message in st.session_state['chat_history']:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🤖"):
            st.markdown(message.content)
    else:
        with st.chat_message("Human", avatar="👤"):
            st.markdown(message.content)


user_query = st.chat_input("💭 Ask me anything about your database... (e.g., How many employees are there?)")
if user_query is not None and user_query.strip() != "":
    if 'db' not in st.session_state:
        st.error("⚠️ Please connect to a database first from the sidebar.")
        st.stop()

    st.session_state["chat_history"].append(HumanMessage(content=user_query))

    with st.chat_message("Human", avatar="👤"):
        st.markdown(user_query)

    with st.chat_message("AI", avatar="🤖"):
        with st.spinner("🔮 Analyzing your query..."):
            response = get_response(
                user_query,
                st.session_state['db'],
                st.session_state['chat_history']
            )

            if response is None or response.strip() == "":
                response = "I am sorry, I could not find an answer to your question."
            st.markdown(response)

    st.session_state["chat_history"].append(AIMessage(content=response))
