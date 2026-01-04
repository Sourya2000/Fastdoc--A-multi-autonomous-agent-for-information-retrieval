# test_chat.py
from langchain_core.messages import HumanMessage


state = {"messages": [HumanMessage(content="What is the main purpose of this PDF?")]}
config = {"configurable": {"thread_id": "thread_123"}}
ground_truth = "This document explains the main purpose of ..."

result = chat_node_with_metrics(state, config=config, ground_truth=ground_truth)

print("Answer:", result["messages"][-1]["content"])
print("Metrics:", result["metrics"])
