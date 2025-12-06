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
    You are an expert SQL database analyst and administrator. You are interacting with a user who is asking questions about the database or requesting data modifications.
    Based on the table schema below, write SQL query/queries that would answer the user's question or perform the requested operation.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    **CRITICAL RULES:**
    1. Write ONLY valid SQL query/queries - no explanations, no backticks, no markdown formatting.
    2. For MULTIPLE operations or questions, write multiple SQL statements separated by semicolons (;).
    3. Always use the exact table and column names from the schema above.
    4. If a table or column doesn't exist in the schema, DO NOT make up names - use only what's available.
    5. For ambiguous questions, infer the most logical table/column based on the schema context.

    **SUPPORTED OPERATIONS:**
    - SELECT: Retrieve data with filtering (WHERE), sorting (ORDER BY), grouping (GROUP BY), joins (JOIN), aggregations (COUNT, SUM, AVG, MIN, MAX), subqueries, DISTINCT, LIMIT, OFFSET
    - INSERT: Add new records (INSERT INTO table (columns) VALUES (values))
    - UPDATE: Modify existing records (UPDATE table SET column=value WHERE condition)
    - DELETE: Remove records (DELETE FROM table WHERE condition)
    - Complex queries: UNION, INTERSECT, EXCEPT, nested subqueries, CTEs (WITH clause), CASE statements
    - Analytical: Window functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD), HAVING clause

    **QUERY WRITING GUIDELINES:**
    - For COUNT/aggregation questions: Use appropriate GROUP BY clauses
    - For "top N" or "bottom N": Use ORDER BY with LIMIT
    - For date/time filtering: Use appropriate date functions (DATE, YEAR, MONTH, etc.)
    - For text search: Use LIKE with wildcards (%) for partial matches
    - For NULL handling: Use IS NULL or IS NOT NULL, COALESCE, IFNULL
    - For joins: Identify relationships from schema and use appropriate JOIN type (INNER, LEFT, RIGHT)
    - For calculations: Use arithmetic operators and aggregate functions
    - Always include WHERE clause for UPDATE/DELETE to prevent affecting all rows (unless explicitly requested)

    **MULTIPLE QUERIES HANDLING:**
    If the user asks multiple questions or requests multiple operations in one prompt:
    - Write each query on a new line, separated by semicolons
    - Process them in logical order (e.g., INSERT before SELECT to see new data)

    **EXAMPLES:**
    Question: Which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;

    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;

    Question: Add a new customer named John with email john@test.com and then show all customers
    SQL Query: INSERT INTO customers (name, email) VALUES ('John', 'john@test.com');
    SELECT * FROM customers;

    Question: Update the price of product with id 5 to 99.99
    SQL Query: UPDATE products SET price = 99.99 WHERE id = 5;

    Question: Delete all orders older than 2020 and count remaining orders
    SQL Query: DELETE FROM orders WHERE YEAR(order_date) < 2020;
    SELECT COUNT(*) as remaining_orders FROM orders;

    Question: Show total sales by category and the top selling product
    SQL Query: SELECT category, SUM(sales) as total_sales FROM products GROUP BY category ORDER BY total_sales DESC;
    SELECT product_name, sales FROM products ORDER BY sales DESC LIMIT 1;

    Question: Find customers who have never placed an order
    SQL Query: SELECT c.* FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.id IS NULL;

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
    You are an expert data analyst and database administrator. You help users understand their database and perform operations on it.
    Based on the information below, provide a clear, helpful, and well-formatted response.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    
    User Question: {question}
    
    SQL Query/Queries Executed: 
    <SQL>{query}</SQL>
    
    SQL Response/Results: 
    {response}

    **RESPONSE GUIDELINES:**

    1. **For SELECT queries (data retrieval):**
       - Present results in a clean, formatted markdown table when applicable
       - Summarize key findings in natural language
       - If multiple queries were executed, present each result separately with clear headings
       - For aggregate results (COUNT, SUM, AVG), highlight the numbers clearly

    2. **For INSERT operations:**
       - Confirm the insertion was successful
       - Mention what data was added
       - If a SELECT follows, show the updated data

    3. **For UPDATE operations:**
       - Confirm the update was successful
       - Specify what was changed and how many rows were affected (if known)
       - Describe the modification made

    4. **For DELETE operations:**
       - Confirm the deletion was successful
       - Mention what was deleted
       - If appropriate, mention how many records were removed

    5. **For multiple queries:**
       - Address each query result in order
       - Use clear section headings (### Query 1 Results, ### Query 2 Results, etc.)
       - Summarize the overall outcome at the end

    6. **Error handling:**
       - If SQL returns no results: "No matching records found for your query."
       - If SQL returns empty data: "The query executed successfully but returned no data."
       - If there's an error: Explain what might have gone wrong in simple terms

    7. **Formatting:**
       - Use markdown tables for tabular data (| Column1 | Column2 |)
       - Use bullet points for lists
       - Bold important numbers and findings
       - Keep responses concise but informative

    **EXAMPLE RESPONSES:**

    For a count query:
    "There are **42 customers** in the database."

    For a data retrieval:
    "Here are the top 5 products by sales:
    
    | Product Name | Sales | Category |
    |--------------|-------|----------|
    | Widget A     | 1500  | Electronics |
    | Widget B     | 1200  | Electronics |
    ..."

    For an insert operation:
    "✅ Successfully added the new customer:
    - **Name:** John Doe
    - **Email:** john@example.com
    
    The customer has been added to the database."

    For multiple operations:
    "### Operation 1: Insert
    ✅ New record added successfully.
    
    ### Operation 2: Current Data
    Here's the updated list:
    | ... |"

    Now provide your response based on the actual query results above:
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")      

    def execute_queries(vars):
        """Execute single or multiple SQL queries separated by semicolons"""
        query = vars['query'].strip()
        results = []
        
        # Split queries by semicolon, but handle edge cases
        queries = [q.strip() for q in query.split(';') if q.strip()]
        
        for i, q in enumerate(queries):
            try:
                result = db.run(q)
                if len(queries) > 1:
                    results.append(f"**Query {i+1}:** `{q}`\n**Result:** {result}")
                else:
                    results.append(result)
            except Exception as e:
                results.append(f"**Query {i+1} Error:** {str(e)}")
        
        return "\n\n".join(results) if results else "No results"

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: execute_queries(vars)
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
