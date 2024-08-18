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
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.document_transformers import Html2TextTransformer
from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.tools import Tool
from langchain_core.messages.base import BaseMessage
import operator
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
warnings.filterwarnings("ignore")


def chatbot_answer (transcript : str,query :str ) :
    text=transcript
    texts = text_splitter.split_text(text)
    text_splitter = CharacterTextSplitter(
    chunk_size=50,
    length_function=len,
    is_separator_regex=False,
    )
    chunked_documents=text_splitter.create_documents([text])
    db= FAISS.from_documents(chunked_documents,HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    retriever=db.as_retriever(
        search_type='similarity',
        search_kwargs={'k':10}
    )
    class VectorStore(BaseModel):
        (
            "A vectorstore contains information about transcript"
            ", summary and caption of youtube video"
        )

        query: str

    router_prompt_template = (
        "You are an expert in routing user queries to a VectorStore\n"
        "The VectorStore contains information on the video.\n"
        'Note that if a query is not video related, you must output "not video related", don\'t try to use any tool.\n\n'
        "query: {query}"
    )
# add llm here chatopenai
    llm = "enter your llm here"
    prompt = ChatPromptTemplate.from_template(router_prompt_template)
    question_router = prompt | llm.bind_tools(tools=[VectorStore])
    class DocumentGrader(BaseModel):
        "check if documents are relevant"

        grade: Literal["relevant", "irrelevant"] = Field(
            ...,
            description="The relevance score for the document.\n"
            "Set this to 'relevant' if the given context is relevant to the user's query, or 'irrlevant' if the document is not relevant.",
        )

        @validator("grade", pre=True)
        def validate_grade(cls, value):
            if value == "not relevant":
                return "irrelevant"
            return value


    grader_system_prompt_template = """"You are a grader tasked with assessing the relevance of a given context to a query. 
        If the context is relevant to the query, score it as "relevant". Otherwise, give "irrelevant".
        Do not answer the actual answer, just provide the grade in JSON format with "grade" as the key, without any additional explanation."
        """

    grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grader_system_prompt_template),
            ("human", "context: {context}\n\nquery: {query}"),
        ]
    )


    grader_chain = grader_prompt | llm.with_structured_output(DocumentGrader, method="json_mode")
    rag_template_str = (
    "You are a helpful assistant. Answer the query below based on the provided context.\n\n"
    "context: {context}\n\n"
    "query: {query}"
)


    rag_prompt = ChatPromptTemplate.from_template(rag_template_str)
    rag_chain = rag_prompt | llm | StrOutputParser()

    fallback_prompt = ChatPromptTemplate.from_template(
    (
        "You are a well renowned youtube analyst your name is 'jarvis'.\n"
        "Do not respond to queries that are not related to the video.\n"
        "If a query is not related to the video, acknowledge your limitations.\n"
        "Provide concise responses to only video related queries.\n\n"
        "Current conversations:\n\n{chat_history}\n\n"
        "human: {query}"
    )
)

    fallback_chain = (
        {
            "chat_history": lambda x: "\n".join(
                [
                    (
                        f"human: {msg.content}"
                        if isinstance(msg, HumanMessage)
                        else f"AI: {msg.content}"
                    )
                    for msg in x["chat_history"]
                ]
            ),
            "query": itemgetter("query") ,
        }
        | fallback_prompt
        | llm
        | StrOutputParser()
    )
    class HallucinationGrader(BaseModel):
        "hallucination grader"

        grade: Literal["yes", "no"] = Field(
            ..., description="'yes' if the llm's reponse is hallucinated otherwise 'no'"
        )


    hallucination_grader_system_prompt_template = (
        "You are a grader assessing whether a response from an llm is based on a given context.\n"
        "If the llm's response is not based on the given context give a score of 'yes' meaning it's a hallucination"
        "otherwise give 'no'\n"
        "Just give the grade in json with 'grade' as a key and a binary value of 'yes' or 'no' without additional explanation"
    )

    hallucination_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", hallucination_grader_system_prompt_template),
            ("human", "context: {context}\n\nllm's response: {response}"),
        ]
    )


    hallucination_grader_chain = (
        RunnableParallel(
            {
                "response": itemgetter("response"),
                "context": lambda x: "\n\n".join([c.page_content for c in x["context"]]),
            }
        )
        | hallucination_grader_prompt
        | llm.with_structured_output(HallucinationGrader, method="json_mode")
    )
    class AnswerGrader(BaseModel):
        "To check if provided answer is relevant"

        grade: Literal["yes", "no"] = Field(
            ...,
            description="'yes' if the provided answer is an actual answer to the query otherwise 'no'",
        )


    answer_grader_system_prompt_template = (
        "You are a grader assessing whether a provided answer is in fact an answer to the given query.\n"
        "If the provided answer does not answer the query give a score of 'no' otherwise give 'yes'\n"
        "Just give the grade in json with 'grade' as a key and a binary value of 'yes' or 'no' without additional explanation"
    )

    answer_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_grader_system_prompt_template),
            ("human", "query: {query}\n\nanswer: {response}"),
        ]
    )


    answer_grader_chain = answer_grader_prompt | llm.with_structured_output(
        AnswerGrader, method="json_mode"
    )
    tool_executor = ToolExecutor(
    tools=[
        Tool(
            name="VectorStore",
            func=retriever,
            description="Useful to search the vector database",
        )
    ]
    )   
    class AgentState(TypedDict):
        """The dictionary keeps track of the data required by the various nodes in the graph"""

        query: str
        chat_history:list[BaseMessage]
        generation: str
        documents: list[Document]


    def retrieve_node(state: dict) -> dict[str, list[Document] | str]:
        """
        Retrieve relevent documents from the vectorstore

        query: str

        return list[Document]
        """
        query = state["query"]
        documents = retriever
        return {"documents": documents}


    def fallback_node(state: dict):
        """
        Fallback to this node when there is no tool call
        """
        query = state["query"]
        chat_history = state["chat_history"]
        generation = fallback_chain.invoke({"query": query, "chat_history": chat_history})
        return {"generation": generation}


    def filter_documents_node(state: dict):
        filtered_docs = list()

        query = state["query"]
        documents = state["documents"]
        for i, doc in enumerate(documents, start=1):
            grade = grader_chain.invoke({"query": query, "context": doc})
            if grade.grade == "relevant":
                print(f"---DOC {i}: RELEVANT---")
                filtered_docs.append(doc)
            else:
                print(f"---DOC {i}: NOT RELEVANT---")
        return {"documents": filtered_docs}


    def rag_node(state: dict):
        query = state["query"]
        documents = state["documents"]

        generation = rag_chain.invoke({"query": query, "context": documents})
        return {"generation": generation}


    def question_router_node(state: dict):
        query = state["query"]
        try:
            response = question_router.invoke({"query": query})
        except Exception:
            return "llm_fallback"

        if "tool_calls" not in response.additional_kwargs:
            print("---No tool called---")
            return "llm_fallback"

        return "VectorStore"

    def should_generate(state: dict):
        filtered_docs = state["documents"]

        if not filtered_docs:
            print("---All retrived documents not relevant---")
            return "llm_fallback"
        else:
            print("---Some retrived documents are relevant---")
            return "generate"


    def hallucination_and_answer_relevance_check(state: dict):
        llm_response = state["generation"]
        documents = state["documents"]
        query = state["query"]

        hallucination_grade = hallucination_grader_chain.invoke(
            {"response": llm_response, "context": documents}
        )
        if hallucination_grade.grade == "no":
            print("---Hallucination check passed---")
            answer_relevance_grade = answer_grader_chain.invoke(
                {"response": llm_response, "query": query}
            )
            if answer_relevance_grade.grade == "yes":
                print("---Answer is relevant to question---\n")
                return "useful"
            else:
                print("---Answer is not relevant to question---")
                return "not useful"
        print("---Hallucination check failed---")
        return "generate"
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(AgentState)
    workflow.add_node("VectorStore", retrieve_node)
    workflow.add_node("filter_docs", filter_documents_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("rag", rag_node)

    workflow.set_conditional_entry_point(
        question_router_node,
        {
            "llm_fallback": "fallback",
            "VectorStore": "VectorStore",
        },
    )

    workflow.add_edge("VectorStore", "filter_docs")
    workflow.add_conditional_edges(
        "filter_docs", should_generate, {"llm_fallback":"fallback" ,"generate": "rag"}
    )
    workflow.add_conditional_edges(
        "rag",
        hallucination_and_answer_relevance_check,
        {"useful": END, "not useful": "fallback", "generate": "rag"},
    )

    workflow.add_edge("fallback", END)
    app = workflow.compile(debug=False)

    response = app.invoke({"query": query, "chat_history": []})
    return (response["generation"])



def main() :
    print("this is the chatbot")

if __name__ == '__main__' :
    main()