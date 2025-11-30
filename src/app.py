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
    1. If user do not specify any table name then assume the table name is 'emp'

    Dangerous Operations:
    - If the SQL query contains any dangerous operations like DELETE, DROP, UPDATE, or ALTER, do not execute the query. Instead, respond with "I'm sorry, but I cannot execute queries that modify or delete data."
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


st.set_page_config("Chat with MySql Database", page_icon='💬')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [
        AIMessage(content="Hello! I'm here to help you, Ask me anything about your data or structure.")
    ]

st.title("💬 Chat with MySql Database")


with st.sidebar:
    st.header("💬 Chat with MySql Database")
    st.markdown("---")

    host = st.text_input("Host", value="sql8.freesqldatabase.com", key='host')
    port = st.text_input("Port", value="3306", key='port')
    user = st.text_input("User", value="sql8809948", key='user')
    password = st.text_input("Password", type="password", value="vFh7Mqw9PP", key='password')
    database = st.text_input("Database", value="sql8809948", key='database')

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(
                    st.session_state['host'],
                    st.session_state['port'],
                    st.session_state['user'],
                    st.session_state['password'],
                    st.session_state['database']
                )
                st.success("Connected successfully!")
                st.session_state['db'] = db
            except Exception as e:
                st.error(f"Connection error: {e}")


for message in st.session_state['chat_history']:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    else:
        with st.chat_message("Human"):
            st.markdown(message.content)


user_query = st.chat_input("eg: How many employees are there in each department?")
if user_query is not None and user_query.strip() != "":
    if 'db' not in st.session_state:
        st.error("Please connect to a database first from the sidebar.")
        st.stop()

    st.session_state["chat_history"].append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(
            user_query,
            st.session_state['db'],
            st.session_state['chat_history']
        )

        if response is None or response.strip() == "":
            response = "I am sorry, I could not find an answer to your question."
        st.markdown(response)

    st.session_state["chat_history"].append(AIMessage(content=response))
