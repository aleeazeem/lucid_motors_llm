from os import name
from re import A
from huggingface_hub import Agent
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
#from langgraph.agents import OrchestratorAgent 
from typing import Dict, TypedDict, List, Any
import dotenv
from sympy import use
dotenv.load_dotenv()


class AgentState(TypedDict):
    """State structure for the orchestrator agent"""
    messages: List[HumanMessage]
    
llm = ChatOpenAI(model="gpt-4", temperature=0)

def process(state: AgentState) -> AgentState:
    """Process the input state and return the output state"""
    response = llm(state['messages'])
    print(f"Response: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter your query: ")
while user_input.lower() not in ["exit", "quit"]:
    agent.invoke({"message": [HumanMessage(content=user_input)]})
    user_input = input("Enter your query: ")    