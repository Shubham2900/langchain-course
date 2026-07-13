from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """ Schema for Response from the agent"""
    answer: str = Field(description="The answer to the question")
    sources: list[Source] = Field(default_factory=list, description="A list of sources used by the agent")

llm = ChatOllama(temprature=0.5, model="gpt-oss:20b")
tools = [TavilySearch()]
agent = create_agent(llm, tools, response_format=AgentResponse)
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
