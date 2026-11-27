import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage #Will invoke pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate  #Will form the RAG prompt we will be using
from langchain_core.runnables import RunnablePassthrough #Runnable that will let the input pass through when it is invoked
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv() #Load env variables

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-6.1-sol")

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

#k=3 to limit to 3 top docs 
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  #as_retriever will give us an object of VectorStoreRetriever - will have search function (searching capabilities) . converts PineconeVectorStore into a retriever object that can perform similarity searches

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)
# {context} will be the augmentation part 

#auxiliary function. docs is langchain document : Take docs that was retrieved from vector store and format them into a nice string
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)  #iterate through docs, take page content and append \n to it. Create one string. 
                                                          #The string created will be sent to {context}


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(  #RunnablePassThrough.assign creates a new dict - combine org input with newly computed field that we're going to explicitely mention
            context=itemgetter("question") | retriever | format_docs   #We do not change input (bcoz RunnablePassThrough) ; dict created with key of question, and value; 
        )
        | prompt_template           #Correspond to step 3,4,5 of earlier implementation.  prompt_template = written above w/ user's question and retrieve context
        | llm                       #pipe to LLM = invoking llm here
        | StrOutputParser()         # to acesss .content of response
    )
    return retrieval_chain

#                        -------------------------------------------------------------------------------
# RunnablePassThrough.assign (.......) : Output is =
#                                           {"question": "What is a PineCone?",     #  Passed through unchanged
#                                             "context": "doc1\n\ndoc2\n\ndoc3"     #  Newly Computed
#                                            } 
#
#                        -------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"

# create_retrieval_chain_with_lcel() ::: this function does not receive any arguments.And the reason for this is that this 
# function is going to eventually return us a LangChain chain.
#So the chain is going to be a Langchain runnable,so we can go and use the invoke method on what this function will return.
#So this is why this function does not receive any input.
#The input to the chain is going to be the input to the invoke function

#itemgetter("abc") is equivalent of lambda expression like ==> lambda x:x['question'] ===> Simply pulls out question string


    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
