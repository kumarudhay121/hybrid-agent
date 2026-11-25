from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_typesafe import TypeSafeClassifier

from hybrid_agent import create_hybrid_agent
from hybrid_agent.routing import JevRouter


@tool
def lookup_employee(employee_id: int) -> str:
    """Look up an employee by their numeric ID."""
    records = {
        101: {"name": "Priya Sharma", "department": "Engineering"},
        102: {"name": "Rahul Verma", "department": "Finance"},
    }
    return str(records.get(employee_id, {"error": "Employee not found"}))


if __name__ == "__main__":
    router = JevRouter(classifier=TypeSafeClassifier())
    executor = ChatOpenAI(model="gpt-4.1", temperature=0)

    agent = create_hybrid_agent(
        router=router,
        executor=executor,
        tools=[lookup_employee],
        max_iterations=5,
    )
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Find employee 101."}]
    })
    print(result["final_answer"])
