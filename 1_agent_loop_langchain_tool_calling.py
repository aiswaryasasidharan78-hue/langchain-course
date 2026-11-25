from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage 
#ToolMessage will contain Tool result
#SystemMessage is to implicate that we are using system message to the LLM
#HumanMessage is for when we use the user input
from langsmith import traceable

MAX_ITERATIONS = 10  #Choose any value above 2
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)
    #Look for product in dictionary "prices" ; if not found return 0


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop ---

#traceable function decorator from Langsmith
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):                              # Agent ReAct loop
    tools = [get_product_price, apply_discount]            #List of tools
    tools_dict = {t.name: t for t in tools}                #Each item in dictioanry is a key:value pair containing key of tool name and value of tool itself
 
    # t in tools (iterating through tools), t.name (key) : t (value)
    # we are creating a dictionary her (tools_dict) because this will help us get the tool name which is the result of LLM, 
    # and the tool itself which is the Python object

    #initialize LLM 
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    # LangChain bind_tools function, and it's going to receive the list of tools and it's going to bind the tools to the model.
    # Every time we're going to be making a request to the LLM, we're also going to be sending the tool descriptions.
    # If LLM supports function calling,then it can return an answer to be a tool call, so which tool to invoke.
    # bind_tools function is going to work for every LLM chat model in LangChain that is going to be supporting function calling.
   
   
    print(f"Question: {question}")
    print("=" * 60)

# ---------------------------------BRAIN OF THE AGENT --------------------------------------------------------------------------

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"     # ----- DEFENSIVE PROMPTING ---------
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    # Continue through iterations until LLM has an answer and not another tool call
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages) #Call LLM with messages, and response AI message is stored. Could be tool call or content

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer. tool_calls is empty
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        #Python LangChain tool which we invoke 
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        # Append messages so that the agent/LLM would have the history and would know what to do
        messages.append(ai_message)
        #Append Tool Result and Tool Call ID -> Feedback to agent
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
