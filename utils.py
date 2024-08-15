import gc
import datetime
import os
import re
from typing import Literal
import streamlit as st
import torch
from diffusers import DiffusionPipeline
import os
from dotenv import load_dotenv
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nest_asyncio
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain.chains.combine_documents import stuff
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from IPython.core.display import Markdown
import json
import re
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnablePassthrough,
)
from langchain_core.pydantic_v1 import validator
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter
import asyncio
import warnings
import PyPDF2
from typing import overload, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.tools import Tool
from langchain_core.messages.base import BaseMessage
import operator

warnings.filterwarnings("ignore")
from keys import key
key_1=key()
your_token=key_1.get_key("HUGGINGFACE_API_KEY")

def generate(
            prompt, 
            pipeline_name,
            command,
            prefix,
            image_input=None,
            mask_input=None,
            negative_prompt=None,
            steps=50,
            width=768,
            height=768,
            guidance_scale=7.5,
            enable_attention_slicing=False,
            enable_cpu_offload=False,
            version="2.1",
            strength=1.0,
            image=None,
             ) :
    if version == "llama-3.1":
        pass
    elif version=="stable-diffusion":
        pass
    elif version == 'aiola/whisper-medusa-v1':
        pass
    elif version == 'openbmb/MiniCPM-Llama3-V-2_5':
        pass



def main():
    print("this is your ustils script")

if __name__ == "__main__":
    main()