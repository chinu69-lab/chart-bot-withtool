%%writefile app.py

import streamlit as st

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LangGraph AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangGraph AI Agent")

st.caption(
    "Tool-calling AI agent with conversation memory"
)


# =========================================================
# 2. SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    # API KEY
    user_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    # PERSONA
    persona = st.text_area(
        "System Prompt / Persona",
        value=(
            "You are a helpful AI assistant. "
            "Use tools whenever they are necessary. "
            "Give clear and accurate answers."
        ),
        height=150
    )

    # RESET
    reset_chat = st.button(
        "🔄 Reset Chat & Apply Persona",
        use_container_width=True
    )

    st.divider()

    st.write("### 🛠️ Available Tools")

    st.write("🧮 Calculator")
    st.write("🌐 Web Search")
    st.write("📝 Word Count")


# =========================================================
# 3. STREAMLIT MEMORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        SystemMessage(content=persona)
    ]


# =========================================================
# 4. RESET CHAT
# =========================================================

if reset_chat:

    st.session_state.messages = [
        SystemMessage(content=persona)
    ]

    st.rerun()


# =========================================================
# 5. DISPLAY PREVIOUS CONVERSATION
# =========================================================

for message in st.session_state.messages:

    # Don't display system prompt
    if isinstance(message, SystemMessage):
        continue

    # User message
    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.markdown(message.content)

    # AI message
    elif isinstance(message, AIMessage):

        # Don't display empty tool-calling AI messages
        if message.content:

            with st.chat_message("assistant"):
                st.markdown(message.content)


# =========================================================
# 6. TOOLS
# =========================================================



from ddgs import DDGS


@tool
def web_search(query: str) -> str:
    """
    Search the web for current and latest information.
    Use this when the user asks about recent information,
    news, prices, events, or information that may have changed.
    """

    print("⭐ Web Search function called")

    try:
        results = DDGS().text(
            query,
            max_results=5
        )

        if not results:
            return "No search results found."

        output = []

        for result in results:
            title = result.get("title", "")
            body = result.get("body", "")
            url = result.get("href", "")

            output.append(
                f"Title: {title}\n"
                f"Description: {body}\n"
                f"URL: {url}"
            )

        return "\n\n".join(output)

    except Exception as e:

        return f"Web search failed: {str(e)}"


@tool
def calculator(a: int, b: int) -> int:
    """
    Adds two integers together.
    Use this tool strictly for addition tasks.
    """

    print("⭐ Calculator function called")

    return a + b


@tool
def get_word_count(text: str) -> int:
    """
    Returns the number of words in the given text.
    """

    print("⭐ Word Count function called")

    return len(text.split())


tools = [
    calculator,
    web_search,
    get_word_count
]


# =========================================================
# 7. LANGGRAPH STATE
# =========================================================

class State(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]


# =========================================================
# 8. CREATE GRAPH
# =========================================================

def create_graph(api_key):

    # -----------------------------------------------
    # Initialize LLM
    # -----------------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=api_key
    )

    # -----------------------------------------------
    # Bind tools
    # -----------------------------------------------

    llm_with_tools = llm.bind_tools(tools)

    # -----------------------------------------------
    # Assistant Node
    # -----------------------------------------------

    def assistant_node(state: State):

        current_messages = state["messages"]

        response = llm_with_tools.invoke(
            current_messages
        )

        return {
            "messages": [response]
        }

    # -----------------------------------------------
    # Tool Node
    # -----------------------------------------------

    tool_node = ToolNode(tools)

    # -----------------------------------------------
    # Routing
    # -----------------------------------------------

    def should_continue(
        state: State
    ) -> Literal["tools", END]:

        messages = state["messages"]

        last_message = messages[-1]

        if last_message.tool_calls:

            return "tools"

        return END

    # -----------------------------------------------
    # Build Graph
    # -----------------------------------------------

    builder = StateGraph(State)

    builder.add_node(
        "assistant",
        assistant_node
    )

    builder.add_node(
        "tools",
        tool_node
    )

    builder.add_edge(
        START,
        "assistant"
    )

    builder.add_conditional_edges(
        "assistant",
        should_continue
    )

    builder.add_edge(
        "tools",
        "assistant"
    )

    return builder.compile()


# =========================================================
# 9. CHAT INPUT
# =========================================================

if user_query := st.chat_input(
    "Ask something..."
):

    # -----------------------------------------------
    # Check API key
    # -----------------------------------------------

    if not user_api_key:

        st.error(
            "Please enter your Groq API key in the sidebar."
        )

        st.stop()


    # =====================================================
    # DISPLAY USER MESSAGE
    # =====================================================

    with st.chat_message("user"):

        st.markdown(user_query)


    # =====================================================
    # ADD USER MESSAGE TO STREAMLIT MEMORY
    # =====================================================

    st.session_state.messages.append(
        HumanMessage(
            content=user_query
        )
    )


    # =====================================================
    # CREATE LANGGRAPH
    # =====================================================

    graph = create_graph(
        user_api_key
    )


    # =====================================================
    # RUN AGENT
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Agent is thinking..."
        ):

            try:

                result = graph.invoke(
                    {
                        "messages":
                        st.session_state.messages
                    }
                )

                # -----------------------------------------
                # Get all new messages generated by Graph
                # -----------------------------------------

                new_messages = result["messages"][
                    len(st.session_state.messages):
                ]

                # -----------------------------------------
                # Store generated messages
                # -----------------------------------------

                for message in new_messages:

                    st.session_state.messages.append(
                        message
                    )

                # -----------------------------------------
                # Find final AI response
                # -----------------------------------------

                final_answer = None

                for message in reversed(
                    st.session_state.messages
                ):

                    if isinstance(
                        message,
                        AIMessage
                    ):

                        if message.content:

                            final_answer = message.content
                            break

                # -----------------------------------------
                # Display final response
                # -----------------------------------------

                if final_answer:

                    st.markdown(
                        final_answer
                    )

                else:

                    st.warning(
                        "The agent did not return a final answer."
                    )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )