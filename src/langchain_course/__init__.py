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

#Here, we are not creating a custom tool, rather we are using the functionality/tool created by Tavily for langchain integration.

#BOOM! HERE IS YOUR AGENT!
llm = ChatOpenAI(model="gpt-5")
tools =[TavilySearch()]
agent = create_agent(model =llm, tools = tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content = "search for 3 job postings for a technical product consultant in UAE on linkedin and list the details")})
    print (result)


if __name__ == "__main__":
    main()

