from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from core.state import AgentState
from core.config import get_llm


def plan_node(state: AgentState) -> dict:
    """Decide what to do. Output a short plan string."""
    llm = get_llm()
    prompt = [
        SystemMessage(content=(
            "You are a task planner for a Databricks knowledge assistant. "
            "The user is asking a question about Databricks. "
            "Produce a 1-sentence plan describing how you will answer the question."
        )),
        HumanMessage(content=state["user_query"]),
    ]
    plan = llm.invoke(prompt).content
    return {"subagent_output": {"plan": plan}}


def execute_node(state: AgentState) -> dict:
    """Execute the plan. Here: single LLM answer. In reality: tool calls, retrieval, etc."""
    llm = get_llm()
    plan = state.get("subagent_output", {}).get("plan", "")
    prompt = [
        SystemMessage(content=(
            "You are a Databricks expert. Answer the user's question about Databricks "
            "clearly and concisely. Follow this plan:\n"
            f"{plan}"
        )),
        HumanMessage(content=state["user_query"]),
    ]
    answer = llm.invoke(prompt).content
    return {"subagent_output": {"plan": plan, "answer": answer}}


def build_subagent():
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)
    return g.compile()
