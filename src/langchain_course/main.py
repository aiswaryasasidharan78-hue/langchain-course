from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama   #:: SUPERLIGHT MODEL

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = '''Avul Pakir Jainulabdeen Abdul Kalam (/ˈʌbdʊl kəˈlɑːm/ ⓘ UB-duul kə-LAHM; 15 October 1931 – 27 July 2015) was an 
    Indian aerospace engineer and science administrator who served as president of India from 2002 to 2007. 
    In 1997, he was awarded India's highest civilian honour, 
    the Bharat Ratna, for his contribution to the scientific research and modernisation of defence technology in India.[2]
    Born and raised in a Muslim family in Rameswaram, Tamil Nadu, Kalam studied physics and aerospace engineering. 
    He spent the next four decades as a scientist and science administrator, mainly at the Defence Research and Development Organisation (DRDO) 
    and Indian Space Research Organisation (ISRO) and was intimately involved in India's civilian space programme and military missile development 
    efforts. He is popularly known as the "Missile Man of India" for his work on the development of ballistic missile and launch vehicle technology. 
    He also played a pivotal organisational, technical, and political role in Pokhran-II nuclear tests in 1998, India's second such test after 
    the first test in 1974. Kalam was elected as the president of India in 2002 with the support of both the ruling Bharatiya Janata Party 
    and the then-opposition Indian National Congress. He was widely referred to as the "People's President". He engaged in teaching, writing 
    and public service after his presidency. While delivering a lecture at IIM Shillong, Kalam collapsed and died from an apparent cardiac 
    arrest on 27 July 2015, aged 83. Thousands attended the funeral ceremony held in his hometown of Rameswaram, where he was buried with full 
    state honours. A memorial was inaugurated near his home town in 2017.''' 



    summary_template = '''
    given the information {information} about a person I want you to create: 
    1. A short summary 
    2. Two interesting facts about them
    '''

    #Create prompt template object, provide input variables. Prompt Template is great for avoid Prompt Injection (rather than using f strings)
    summary_prompt_template = PromptTemplate(
        input_variables = ["information"], template = summary_template
    )

    #Will make the API calls to Open AI . Langchain ChatOpenAI object which under the hood will use the Open AI SDK to make API calls to Open AI with th API key
    #  Initialize LLM
    llm = ChatOpenAI(temperature = 0, model = "gpt-5")
    #Alternate model using Ollama
    #llm = ChatOllama(temperature = 0, model = "gemma3:270m")
    #chain 1 ; Create chain
    chain = summary_prompt_template | llm
    #Execute chain
    response = chain.invoke(input = {"information":information})
    #Display response
    print(response.content)

if __name__ == "__main__":
    main()
