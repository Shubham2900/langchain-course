from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool Searches over the internet
    :param query: the query to search for
    :return: The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query)

llm = ChatOllama(temprature=0.5, model="gpt-oss:20b")
tools = [search]
agent = create_agent(llm, tools)
def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="What's the weather in Tokyo?")
            ]
        }
    )
    print(result)

if __name__ == "__main__":
    main()
