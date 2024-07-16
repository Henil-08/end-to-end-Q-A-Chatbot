import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os

import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Assistant. Please respond to the user queries"),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, engine, temperature, max_tokens, api_key=None):
    if api_key is not None:
        openai.api_key=api_key
        llm=ChatOpenAI(model=engine, temperature=temperature, max_tokens=max_tokens)
    else:
        llm=ChatGroq(model=engine, temperature=temperature, max_tokens=max_tokens)
    output_parser=StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question': question})
    return answer


## #Title of the app
st.title("Enhanced Q&A Chatbot With Latest LLM Models")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Open AI API Key:", type="password")

## Select the OpenAI model
engine=st.sidebar.selectbox("Select Open AI model", ["gpt-4o","gpt-4-turbo","gpt-4", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it", "gemma-7b-it", "mixtral-8x7b-32768"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=5000, value=500)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

openai_engines = ["gpt-4o","gpt-4-turbo","gpt-4"]

if engine in openai_engines:
    if user_input and api_key:
        response=generate_response(user_input,engine,temperature,max_tokens,api_key=api_key)
        st.write(response)
    elif user_input:
        st.warning("Please enter the OpenAI API Key in the slider bar")
    else:
        st.write("Please provide the user input")
else:
    if user_input:
        response=generate_response(user_input,engine,temperature,max_tokens)
        st.write(response)
    else:
        st.write("Please provide the user input")
    
