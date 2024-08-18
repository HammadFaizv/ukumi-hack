import gc
import datetime
import os
import re
from typing import Literal
import streamlit as st
import torch
from diffusers import DiffusionPipeline
import os
import matplotlib.pyplot as plt

from test_bot2 import generate_response
from transcript_gen import transcript_gen
# from test_bot import chat_bot
# from dotenv import load_dotenv
# from langchain_community.document_loaders.web_base import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import nest_asyncio
# from langchain_community.vectorstores.chroma import Chroma
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
# from langchain.chains.combine_documents import stuff
from operator import itemgetter
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# from IPython.core.display import Markdown
import json
import re
# from langchain_core.runnables import (
    # RunnableParallel,
    # RunnableBranch,
    # RunnablePassthrough,
# )
# from langchain_core.pydantic_v1 import validator
# from langchain_core.messages import HumanMessage, AIMessage
# from operator import itemgetter
# import asyncio
import warnings
# import PyPDF2
# from typing import overload, Optional
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_groq import ChatGroq
# from sentence_transformers import SentenceTransformer
# from typing import TypedDict, Annotated
# from langchain_core.documents import Document
# from langgraph.prebuilt import ToolInvocation, ToolExecutor
# from langchain_core.tools import Tool
# from langchain_core.messages.base import BaseMessage
import operator

from eval import evaluation
from test_thumbnail import generate_summary, generate_caption, generate_image

def chat_bot():
    transcript = ""
    with open("temp/trans.txt", "r") as file:
                transcript = file.read()
            
    st.title("Chatbot")

            # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

            # User input
    user_input = st.text_input("You: ", "")

    if user_input:
        with st.spinner("Generating ..."):
            st.session_state['chat_history'].append(f"You: {user_input}")

            # Generate response
            response = generate_response(transcript, user_input)
            st.session_state['chat_history'].append(f"Bot: {response}")

    # Display chat history
    for message in st.session_state['chat_history']:
        st.write(message)

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
    elif version == 'deepgram_whisper':
        st.video(prompt)
        res = transcript_gen(prompt)
        
        transcript = res['results']['channels'][0]['alternatives'][0]['transcript']
        # Extracting sentences and their timestamps
        sentences = []

        for channel in res['results']['channels']:
            for alternative in channel['alternatives']:
                for paragraph in alternative['paragraphs']['paragraphs']:
                    for sentence in paragraph['sentences']:
                        start_time = sentence['start']
                        end_time = sentence['end']
                        text = sentence['text']
                        sentences.append((start_time, end_time, text))

        st.subheader("Transcript: ")
        for start, end, sentence in sentences:
            st.write(f"**Start:** {start:.2f}s, **End:** {end:.2f}s")
            st.write(f"{sentence}")
            st.write("---")

        # write transcript to save
        with open("temp/trans.txt", "w") as file:
            file.write(transcript)

        summary = generate_summary(transcript)
        st.subheader("Summary: ")
        st.write(summary)

        with open("temp/sums.txt", "w") as file:
            file.write(summary)

    elif version == 'caption_generator':
        summary = ""
        with open("temp/sums.txt", "r") as file:
            summary = file.read()
        
        caption = generate_caption(summary)
        st.subheader("Caption: ")
        st.write(caption)

        with st.spinner("Generating image..."):
            image_url = generate_image(summary)
            st.subheader("Thumbnail: ")
            st.image(image_url, caption="Generated Image")

    elif version == 'chatbot':

        transcript = ""
        with open("temp/trans.txt", "r") as file:
            transcript = file.read()
            

            # Chat history
        if 'chat_history' not in st.session_state:
            # print(1)
            st.session_state['chat_history'] = []

        # User input
        user_input = prompt

        if user_input:
            with st.spinner("Generating ..."):
                # print(2)
                st.session_state['chat_history'].append(f"You: {user_input}")

                # Generate response
                response = generate_response(transcript, user_input)
                st.session_state['chat_history'].append(f"Bot: {response}")

        # Display chat history
        st.subheader("Chatbot")
        for message in st.session_state['chat_history']:
            st.write(message)
        # print(3)

    elif version == 'eval':
        transcript = ""
        with open("temp/trans.txt", "r") as file:
            transcript = file.read()
        
        result = evaluation(transcript)

        metrics = [i for i in result.keys() if i not in ['COMPLEX WORD COUNT', 'WORD COUNT', 'PERSONAL PRONOUNS']]
        values = [result[key] for key in metrics]

        values.append(values[0])
        metrics.append(metrics[0])

        # Calculate angle of each metric
        angles = [n / float(len(metrics)-1) * 2 * 3.14159 for n in range(len(metrics))]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Draw the polygonal graph
        ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid')
        ax.fill(angles, values, color='blue', alpha=0.25)

        # Add the metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics[:-1], size=12)

        # Display the polygonal graph
        plt.title('Polygonal Graph of Text Analysis Metrics', size=15, color='blue', y=1.1)
        st.pyplot(fig)
        
        if 'eval_history' not in st.session_state:
            st.session_state['eval_history'] = []

    
            for val in result:
                st.session_state['eval_history'].append(f"{val} :  {result[val]}")

        st.subheader("Evaluation: ")
        for message in st.session_state['eval_history']:
            st.write(message)

    

def main():
    print("this is your ustils script")

if __name__ == "__main__":
    main()