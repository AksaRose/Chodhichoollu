from typing import Literal
import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, List, TypedDict
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage,HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import store
import re


load_dotenv()

os.getenv("LANGSMITH_TRACING")
os.getenv("LANGSMITH_API_KEY")
os.getenv("GOOGLE_API_KEY")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = store.vectorstore.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state:MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return({"messages" : [response]})

# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = f"""You are a helpful assistant for question-answering tasks. You have access to both:

    1. CONVERSATION HISTORY: Pay attention to the previous conversation to maintain context and remember what the user has told you.

    2. RETRIEVED DOCUMENTS: Use the following retrieved context when relevant to answer questions about the documents.

    Retrieved Context:
    {docs_content}

    Instructions:
    - Always consider the conversation history when answering
    - If the user asks about something mentioned earlier in the conversation, refer to that information
    - If the question requires information from the documents, use the retrieved context
    - If you don't know the answer from either source, say that you don't know
    - Keep answers concise but informative in 3 or 4 lines at maximum
    - Maintain a conversational tone and remember previous interactions"""
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    def clean_spaced_latex(text: str) -> str:
        """
        Remove unnecessary spaces inside LaTeX commands and formulas.
        """
        # Remove spaces after backslash in commands: \text { -> \text{
        text = re.sub(r'\\([a-zA-Z]+)\s*{', r'\\\1{', text)

        # Remove spaces around operators inside math mode
        def remove_inner_spaces(match):
            content = match.group(1)
            # remove spaces between letters/numbers/operators inside formula
            content = re.sub(r'\s+', '', content)
            return f'${content}$'  # keep single $ for inline or double $$ for block

        # Apply only to block formulas $$...$$
        text = re.sub(r'\$\$(.+?)\$\$', lambda m: f'\n$$\n{m.group(1).replace(" ", "")}\n$$\n', text, flags=re.DOTALL)

        return text
    raw_response = llm.invoke(prompt)
    # Extract string content
    if hasattr(raw_response, "content"):
        text = raw_response.content
    else:
        text = str(raw_response)

    # Auto convert formulas to block equations
    text = clean_spaced_latex(text)

    # Return as AIMessage
    return {"messages": [AIMessage(content=text)]}




graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

