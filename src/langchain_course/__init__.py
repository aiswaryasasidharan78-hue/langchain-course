'''
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
#Importing the tool decorator
from langchain_core.messages import HumanMessage
#Will use Human Message to invoke agent
from langchain_openai import ChatOpenAI
#Will give agent LLM
from tavily import TavilyClient


tavily = TavilyClient()

@tool
def search(query : str) -> str: 
#DOC STRINGS ::: VERY IMPORTANT
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Returns: 
        The search result
    """
    print (f"Searching for {query}")
    #return "Tokyo weather is Sunny"
    return tavily.search(query = query)

#BOOM! HERE IS YOUR AGENT!
llm = ChatOpenAI(model="gpt-5")
tools =[search]
agent = create_agent(model =llm, tools = tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content = "search for 3 job postings for a technical product consultant in UAE on linkedin and list the details")})
#Sending Human Message along with the content "What is the weather in Tokyo"
#The key here is messages, but we did not provide list of messages, rather a single message (Human Message), not inside a list
#This is okay because under the hood Langchain will convert this into a list of single element
    print (result)


if __name__ == "__main__":
    main()

'''

#----------------------------------------------------------------------------------------------------------------------------------
# ALTERNATE USING TAVILY TOOL COMPOSED BY TAVILY TEAM WITH LANGCHAIN-TAVILY IMPORT
#----------------------------------------------------------------------------------------------------------------------------------
from typing import List
from pydantic import BaseModel, Field
#Pydantic's BaseModel will give us a Base Class, from which we can inherit a structred data schema
#We get functionalities like data parsing, serialization, automatic type validation 
#Field class allows to add metadata to our model's attributes. So we can add descriptions which is going to be helpful for the LLMs to understand what to put in that field.

from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
#Importing the tool decorator
from langchain_core.messages import HumanMessage
#Will use Human Message to invoke agent
from langchain_openai import ChatOpenAI
#Will give agent LLM
from langchain_tavily import TavilySearch


#Describe sources for agent to get answers
class Source (BaseModel):
    """Schema for a source used by an agent"""
#Defining a field called url, which is going to be a string
    url:str = Field(description="URL of the source")
#Point of this class is to represent the source of the answer




#This source Pydantic class to be a nested field in another Pydantic class,which is going to be called agent response.
#So agent response is going to have an answer and it's going to have a list of sources.


class AgentResponse(BaseModel):
    """Schema for Agent Response with answer and sources"""
    answer:str = Field(description= "The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list,description="List of sources used to generate the answer" )
# default factory: is going to be list.This will make that when we create an agent response class,if we don't provide the sources,
# then the sources attribute is going to be populated with an empty list.

#Here, we are not creating a custom tool, rather we are using the functionality/tool created by Tavily for langchain integration.
#BOOM! HERE IS YOUR AGENT!
llm = ChatOpenAI(model="gpt-5")
tools =[TavilySearch()]
agent = create_agent(model =llm, tools = tools, response_format=AgentResponse)
#Adding a response format value will tell the agent the exact format we want our answers in, and not the string we used to get


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content = "search for 3 job postings for a technical product consultant in UAE on linkedin and list the details")})
    print (result)


if __name__ == "__main__":
    main()

