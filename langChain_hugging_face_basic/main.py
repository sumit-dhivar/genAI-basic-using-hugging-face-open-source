#pip install langchain huggingface_hub
#pip install langchain_community
import os
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
secretKey = os.getenv('HUGGING_FACE_FLAN_API_KEY')
# Set up your Hugging Face API key
huggingface_api_key = secretKey

# Initialize HuggingFaceHub LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",  # Smaller, faster model
    huggingfacehub_api_token=huggingface_api_key,
)

# Define a prompt template
template = "Translate the English to French: {text}"
prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

# Create a Langchain LLM chain with the model and the prompt
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Run the LLM chain
result = llm_chain.run("my name is sumit!")
print(result)
# This model might not give the correct output for some prompts, 
# because it uses a smaller version having less parameters. 
# To overcome this we can use lager model having higher number of parameters.