from typing import TypedDict
from langgraph.graph import END, StateGraph


class State(TypedDict):
    number: int


def double(state: State) -> dict:
    board_number = state["number"]
    new_number = board_number * 2
    print(f"Double: {new_number}")
    return {"number": new_number}


def finish(state: State) -> dict:
    finish_num = state["number"]
    print(f"Finish number is: {finish_num}")
    return {
        "number":finish_num
    }
def decision(state:State)-> str:
    if state ["number"]<100:
        return "double"
    else:
        return "finish"


builder = StateGraph(State)
builder.add_node("double",double)
builder.add_node("finish",finish)

# Entry point
builder.set_entry_point("double")
builder.add_conditional_edges(
    "double",
    decision,
    {"double":"double","finish":"finish"},

)
builder.add_edge("finish",END)
graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"number":5})
