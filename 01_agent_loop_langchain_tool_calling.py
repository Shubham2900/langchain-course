from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gpt-oss:20b"

# ----Tools (LangChain Tool Decorator) ---

@tool()
def get_product_price(product: str) -> float:
    """ Returns the price of a given product from the Catalog"""
    print(f">> Executing get_product_price for {product}")
    prices = {
        "Laptop": 1299.99,
        "Camera": 12.99,
        "Computer": 199.99,
        "Desktop": 12939.99,
        "Mouse": 19.99,
    }
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str, product: str) -> float:
    """"Apply discount to price. Available tier bronze, silver, gold"""
    print(f">> Executing apply_discount for {product} for discount tier {discount_tier}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temprature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("="*60)

    messages = [
        SystemMessage(
            "You are a helpful Shopping Assistant."
            " You have access to a product catalog tool"
            " and a discount tool. \n\n"
            "STRICT RULES - you must fllow these exactly: \n"
            "1. Never guess or assume a product price"
            "You must call get_product_price(product) to get the price"
            "2. Only call apply_discount(price, tier, product) Adter "
            "you have recieved a price from get_product_price(product)."
            "Pass the exact Price- Do not pass a made-up price"
            "3. nEVER CALCULATE DISCOUNTS YURSELF USING MATH."
            "Always use apply_discount(price, tier, product)"
            "4. If user does nt pass a discount tier. ASK THEM"
            "Don't assume a discount tier."
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n ------------Iteration: {iteration} ------------")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(F"Tool {tool_name} is not available")

        observation  = tool_to_use.invoke(tool_args)
        print(f"Observation: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Error: Max iterations reached")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")



















if __name__ == "__main__":
    print("Hello Langchain Agent (.bind tools)!")
    print()
    result = run_agent("What is the price of a Laptop with Gold Discount?")