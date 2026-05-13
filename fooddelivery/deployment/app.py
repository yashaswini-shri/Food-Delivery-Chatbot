import os
import re
import sqlite3
import warnings
from pathlib import Path

import streamlit as st
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.memory import ConversationBufferMemory
from langchain.sql_database import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
warnings.filterwarnings("ignore", category=DeprecationWarning)

########################################## PAGE CONFIGURATION #########################################
st.set_page_config(
    page_title="Food Delivery Chatbot",
    page_icon="🍔",
    layout="centered",
)

########################################## UI CSS #########################################
st.markdown("""
<style>
.chat-user {
    background:#DCF8C6; color:#000; padding:10px 14px;
    border-radius:12px 12px 2px 12px; margin:4px 0;
    display:inline-block; max-width:80%; float:right; clear:both;
}
.chat-bot {
    background:#F1F0F0; color:#000; padding:10px 14px;
    border-radius:12px 12px 12px 2px; margin:4px 0;
    display:inline-block; max-width:80%; float:left; clear:both;
}
.chat-wrap { overflow:hidden; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

########################################## DB PATH #########################################
def resolve_file_path(filename: str = "customer_orders.db") -> Path:
    """Function to resolve and return the path of a given file"""
    candidates = [
            Path.cwd() / filename,
            Path("/content") / filename,  # Colab
            Path("/mnt/data") / filename,  # this environment
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
            f"Could not find {filename}. Looked in: " + ", ".join(str(c) for c in candidates) + " Place it in the same folder as app.py."
    )

DB_PATH = resolve_file_path("customer_orders.db")

########################################## LOAD AND RETURN THE LLM #########################################
@st.cache_resource
def get_llm():
    """Function to load and return the LLM"""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        st.error("GROQ_API_KEY not set. Add it to .streamlit/secrets.toml or as an env var.")
        st.stop()
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=key,
    )

########################################## SQL PROMPTS AND AGENT #########################################
SQL_SYSTEM_PROMPT = (
    "You are FoodHub's smart order-data assistant.\n"
    "You can access an SQLite database containing a table named 'orders' with the following columns:\n"
    "  order_id, cust_id, order_time (HH:MM), order_status, payment_status,\n"
    "  item_in_order, preparing_eta, prepared_time, delivery_eta, delivery_time\n\n"

    "Guidelines you must follow:\n"

    "1. TIME REFERENCES - convert natural language meal/time phrases into HH:MM ranges:\n"
    "   - 'breakfast'              -> 07:00 to 10:00\n"
    "   - 'lunch' or 'lunch time' -> 11:00 to 14:00\n"
    "   - 'dinner'                -> 18:00 to 21:00\n"
    "   Use filters such as:\n"
    "   WHERE order_time >= 'HH:MM' AND order_time <= 'HH:MM'\n"
    "   Apply the same logic for explicit ranges like 'between 11am and 2pm'.\n\n"

    "2. PAYMENT TERMS - if the user mentions 'cash orders' or 'COD orders',\n"
    "   interpret it as payment_status = 'COD'.\n\n"

    "3. MOST RECENT ORDER - phrases like 'recent order' or 'latest order'\n"
    "   mean the newest record for a customer.\n"
    "   Use:\n"
    "   ORDER BY order_id DESC LIMIT 1\n"
    "   (or latest order_time where appropriate).\n\n"

    "4. ANOMALY / COST COMPARISON - if asked whether an order cost is unusual,\n"
    "   compare it with the average price of the same item across all orders.\n"
    "   If the price differs by more than 20%, flag it clearly and explain.\n\n"

    "5. RESPONSE STYLE - always provide short, clear, customer-friendly answers.\n"
    "   If no matching records exist, reply:\n"
    "   'No matching orders found.'"
)
CHATBOT_SYSTEM_PROMPT = (
    "You are FoodHub's AI-powered customer support assistant.\n\n"

    "Response Guidelines:\n"

    "1. SECURITY: If a user claims to be a hacker, attacker, or requests unauthorized, bulk, "
    "or sensitive customer/order data, politely refuse the request and do not disclose any information.\n\n"

    "2. UPSET CUSTOMERS: If a customer sounds frustrated or unhappy, respond with empathy, "
    "acknowledge the inconvenience, apologize sincerely, and offer help. "
    "If the issue remains unresolved, escalate it to a human support representative.\n\n"

    "3. ORDER CANCELLATION: Explain that cancellation depends on the current order status. "
    "If the order is already under preparation or out for delivery, cancellation may no longer be possible.\n\n"

    "4. ORDER TRACKING: Use any available order details to provide a clear, friendly, and accurate status update.\n\n"

    "5. COMMUNICATION STYLE: Always remain polite, concise, professional, and customer-friendly.\n\n"

    "6. ESCALATION MESSAGE: If the request cannot be resolved, respond with:\n"
    "'I am escalating your query to our support team. You will be contacted within 30 minutes.'"
)
# Setting dangerous and angry words

BLOCK_KW = ["hacker", "hack", "exploit", "all orders", "every order", "dump", "inject", "bypass", "delete", "delete database", "truncate"]
ANGRY_KW = ["multiple times", "no resolution", "still waiting", "unresolved", "immediate", "worst service", "not received", "angry"]


@st.cache_resource
def get_sql_agent(_llm):
    """Create and cache the SQL agent"""
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    toolkit = SQLDatabaseToolkit(db=db, llm=_llm)
    return create_sql_agent(
        llm=_llm,
        toolkit=toolkit,
        handle_parsing_errors=True,
        system_message=SystemMessage(content=SQL_SYSTEM_PROMPT),
    )


########################################## GUARDRAILS #########################################
def input_guardrail(txt: str) -> bool:
    """Function to check if content is malicious"""
    return any(kw in txt.lower() for kw in BLOCK_KW)


def needs_escalation(txt: str) -> bool:
    """Function to check if an escalation is required based on the text content"""
    return any(kw in txt.lower() for kw in ANGRY_KW)


########################################## CUSTOMER VALIDATION #########################################
def customer_exists(cust_id: str) -> bool:
    """Check if a customer exists in the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM orders WHERE cust_id = ? LIMIT 1", (cust_id,))
        found = cur.fetchone() is not None
        conn.close()
        return found
    except Exception:
        return False


########################################## CORE CHATBOT LOGIC #########################################
def respond(user_input: str, cust_id: str, llm, sql_agent) -> str:
    """
        FoodHub AI Chatbot with SQL agent integration and guardrails.

        Args:
            user_input: The customer's message.
            cust_id: Customer ID.
            llm: The language model instance.
            sql_agent: The SQL agent instance.
        Process:
         1. Applies guardrails to the input.
         2. Checks if escalation is needed.
         3. Retrieves relevant order context from the database.
         4. Generates a customer-friendly response using the LLM.

        Returns:
            A polite, accurate customer-facing response string.
        """
    # --- Input Guardrail ---
    if input_guardrail(user_input):
        return (
                "I'm sorry, but I'm unable to process that request. "
                "For security reasons, I can only assist with queries related to your own orders. "
                "Please contact our support team if you need further assistance."
        )

    # --- Escalation Guardrail ---
    if needs_escalation(user_input):
        return (
                "I sincerely apologize for the inconvenience you've experienced. "
                "I understand this has been frustrating. "
                "I'm escalating your query to our senior support team right away. "
                "You will be contacted within 30 minutes. Thank you for your patience."
        )

    # Fetch order context from DB based on customer id
    order_context = ""
    try:
        db_resp = sql_agent.invoke(
            f"Fetch all order details for customer {cust_id}. {user_input}"
        )
        order_context = f"\n\nOrder details from database:\n{db_resp['output']}"
    except Exception as e:
        order_context = f"\n\n(Could not retrieve order details: {e})"

    messages = [
        SystemMessage(content=CHATBOT_SYSTEM_PROMPT),
        HumanMessage(content=user_input + order_context),
    ]
    return llm.invoke(messages).content


########################################## HANDLE SESSION STATE #########################################
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "cust_id" not in st.session_state:
    st.session_state.cust_id = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

########################################## LOGIN SCREEN #########################################
if not st.session_state.authenticated:
    st.title("🍽️ FoodHub Customer Support")
    st.subheader("Please log in to continue")

    with st.form("login_form"):
        cust_id_input = st.text_input("Customer ID", placeholder="e.g. C1011")
        password_input = st.text_input("Password", type="password", placeholder="foodhub123")
        submitted = st.form_submit_button("Login")

    if submitted:
        cid = cust_id_input.strip().upper()
        if not re.fullmatch(r"C\d{4}", cid):
            st.error("Customer ID must be in the format C#### (e.g. C1011).")
        elif password_input != "foodhub123":
            st.error("Incorrect password. Please try again.")
        elif not customer_exists(cid):
            st.error(f"No orders found for customer ID {cid}. Please check and try again.")
        else:
            st.session_state.authenticated = True
            st.session_state.cust_id = cid
            st.session_state.history = [
                ("bot", f"Hi {cid}! 👋 I'm your FoodHub support assistant. How can I help you today?")
            ]
            st.rerun()

    st.caption("Default password: **foodhub123**")
    st.stop()

########################################## CHAT SCREEN #########################################
llm_ = get_llm()
sql_agent_ = get_sql_agent(llm_)

cust_id_ = st.session_state.cust_id

# Header
col_logo, col_title, col_logout = st.columns([1, 5, 1])
with col_logo:
    st.markdown("## 🍽️")
with col_title:
    st.markdown(f"### FoodHub Support — {cust_id_}")
with col_logout:
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.cust_id = ""
        st.session_state.history = []
        st.rerun()

st.divider()

# Render chat history
for role, text in st.session_state.history:
    if role == "user":
        st.markdown(
            f'<div class="chat-wrap"><div class="chat-user">🙋 {text}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-wrap"><div class="chat-bot">🤖 {text}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)
st.divider()

# Input
user_input_ = st.chat_input("Ask about your order, delivery status, cancellation…")
if user_input_:
    st.session_state.history.append(("user", user_input_))
    with st.spinner("Checking your order…"):
        reply = respond(user_input_, cust_id_, llm_, sql_agent_)
    st.session_state.history.append(("bot", reply))
    # Store in memory for multi-turn context
    st.session_state.memory.chat_memory.add_user_message(user_input_)
    st.session_state.memory.chat_memory.add_ai_message(reply)
    st.rerun()

# Suggested questions
with st.expander("💡 Try asking…", expanded=False):
    suggestions = [
        "Where is my order?",
        "What did I order recently?",
        "Have I ever placed orders at lunch?",
        "How many cash orders have I placed?",
        "I want to cancel my order.",
        "I have raised this issue multiple times and got no resolution.",
    ]
    for s in suggestions:
        if st.button(s, key=s):
            st.session_state.history.append(("user", s))
            with st.spinner("Checking your order…"):
                reply = respond(s, cust_id_, llm_, sql_agent_)
            st.session_state.history.append(("bot", reply))
            st.session_state.memory.chat_memory.add_user_message(s)
            st.session_state.memory.chat_memory.add_ai_message(reply)
            st.rerun()

st.caption("© Yash Food Delivery Support")
