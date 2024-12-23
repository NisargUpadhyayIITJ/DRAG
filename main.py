import os
import openai
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor
from src.reranker import jina_reranker
from src.router import get_routed_agent
from chromaDB.chromadb_client import ChromaRetriever
from src.dynamic_crew import get_dynamic_agents
from src.config import (
    META_AGENT_PROMPT,
    META_AGENT_TASK,
    MARKDOWN_TASK,
    QUE_ANS_AGENT_PROMPT,
    QUE_ANS_AGENT_TASK,
    ALLOW_DELEGATION,
    MODEL
)
from src.crewai_reflection import reflection
from src.crewai_multi import get_multi_agents
from src.guardrail import ( 
    guardrails,
    query_refinment
)
from crewai import (
    Agent,
    Task,
    Crew,
    Process
)
from typing import List
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

chroma_retriever = ChromaRetriever(port=5001)
openai.api_key = os.getenv("OPENAI_API_KEY_GHAR")

def get_collection_name(file_name: str) -> str:
        collection_name = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
        collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
        collection_name = collection_name.strip("_-")

        if len(collection_name) > 63:
            collection_name = collection_name[:61] + "A"
        logger.debug(f"Collection name: {collection_name}")
        return collection_name


def get_context(
    collection_name: str,
    query: str,
    topk: int,
    reranker: bool=False,
    method: str="cr"
)-> List[str]:

    if method == "cr":
        retrieved_docs = chroma_retriever.adv_retrieve_documents(query=query, collection_name=collection_name, n_results=topk*6)
    elif method == "hs":
        retrieved_docs = chroma_retriever.adv_retrieve_documents(query=query, collection_name=collection_name, n_results=topk*6)
    elif method == None:
        retrieved_docs = chroma_retriever.return_final_retrieve_docs(query=query, collection_name=collection_name, n_results=topk*6)
    else:
        raise ValueError(f"Invalid method: {method}")
    
    logger.trace(f"Retrieved {len(retrieved_docs)} documents")
    
    if reranker == True:
        retrieved_docs = jina_reranker(
            query=query,
            documents=retrieved_docs,
            topk=topk
        )
    else:
        retrieved_docs = retrieved_docs[:topk]

    logger.debug(f"Retrieved {len(retrieved_docs)} documents after reranking")

    context = " ".join(retrieved_docs)

    return context


def pipeline(
    query: str,
    collection_name: str,
    topk: int,
    reranker: bool=True,
    method: str="cr", 
    agent_type: str = "dynamic",
    use_reflection: bool=True,
    n_reflection: int=2, 
    use_router: bool=True
)-> str:
    
    passed = guardrails(query)

    logger.info(f"Query after guardrails: {query}")
    if passed == False:
        return "Query has toxic language"
    
    query = query_refinment(query)

    with ThreadPoolExecutor() as executor:
        context_future = executor.submit(
            get_context,
            collection_name=collection_name,
            query=query,
            topk=topk,
            reranker=reranker,
            method=method
        )
        
        if agent_type == "dynamic":
            # Submit the document context retrieval
            doc_context_future = executor.submit(
                chroma_retriever.get_document_context,
                collection_name=collection_name
            )
            
            # Chain `get_dynamic_agents` to the result of `doc_context_future`
            dynamic_agents_future = executor.submit(
                get_dynamic_agents,
                document=doc_context_future.result()
            )
        else:
            dynamic_agent_list = []
            dynamic_tasks_list = []
            static_agent_list, static_tasks_list =get_multi_agents
        

        context = context_future.result()

    que_ans_agent = Agent(
        role=QUE_ANS_AGENT_PROMPT["role"],
        goal=QUE_ANS_AGENT_PROMPT["goal"],
        backstory=QUE_ANS_AGENT_PROMPT["backstory"],
        verbose=QUE_ANS_AGENT_PROMPT["verbose"],
        allow_delegation=ALLOW_DELEGATION,
        llm=MODEL
    )

    que_ans_task_description = QUE_ANS_AGENT_TASK["description"].format(query, context)

    que_ans_task = Task(
        description=que_ans_task_description,
        expected_output=QUE_ANS_AGENT_TASK["expected_output"],
        agent=que_ans_agent,
        async_execution=True
    )

    static_agent_list, static_tasks_list = [], []

    if agent_type == "dynamic":
            dynamic_agent_list, dynamic_tasks_list = dynamic_agents_future.result()
            static_agent_list = [que_ans_agent]
            static_tasks_list = [que_ans_task]

    if agent_type!="dynamic" and agent_type!="multi":
        raise ValueError(f"Invalid agent type: {agent_type} --> choose one of 'dynamic' or 'multi'")

    if use_router and len(dynamic_agent_list)>0:
        dynamic_agent_list, dynamic_tasks_list = get_routed_agent(
            agent_list=dynamic_agent_list,
            task_list=dynamic_tasks_list,
            query=query
        )
    
    agent_list = static_agent_list + dynamic_agent_list
    tasks_list = static_tasks_list + dynamic_tasks_list

    meta_agent= Agent(
            role= META_AGENT_PROMPT["role"],
            goal= META_AGENT_PROMPT["goal"],
            backstory= META_AGENT_PROMPT["backstory"],
            verbose=True,
            allow_delegation= META_AGENT_PROMPT["allow_delegation"]
        )

    meta_agent_task_description = META_AGENT_TASK["description"].format(que=query)

    meta_agent_task = Task(
            description=meta_agent_task_description,
            expected_output=META_AGENT_TASK["expected_output"],
            agent=meta_agent,
            context= tasks_list,
            async_execution=False
        )

    if use_reflection:
        response = reflection(
            query=query,
            context=context,
            agents=agent_list,
            agents_task=tasks_list,
            meta_agent=meta_agent,
            meta_agent_task=meta_agent_task,
            n=n_reflection).raw
        
    else:
        meta_agent_task.expected_output = MARKDOWN_TASK
        initial_crew = Crew(
            agents= [*agent_list, meta_agent],
            tasks= [*tasks_list, meta_agent_task],
            process=Process.sequential,
            verbose=True,
            knowledge={"correct_context": [context], "metadata": {"preference": "personal"}},
            memory=False,
            output_log_file="./logs/log_text.md"
        )

        inputs = {
            'que': query,
            'con': context
        }
        
        response = initial_crew.kickoff(inputs=inputs).raw
        logger.info(f"response without guardrail--{response}\nfor the query:\n{query}")

    passed = guardrails(response)

    if passed is True:
        logger.info (f"Final response after final guardrail:\n {response} \nfor the query:\n{query}")
        return response
    
    else:
        return "Sorry the response contains sensitive content. Please re-enter the query if the error still perssists check the document content."


def main():
    query = "Write about independent investigation as mentioned in the document?"
    pdf_path="./uploaded_files/Nisarg_report.pdf"
    pdf_name = pdf_path.split('/')[-1]
    collection_name = get_collection_name(pdf_name)
    
    pipeline(
        query=query,
        collection_name=collection_name,
        topk=5,
        reranker=True,
        method="cr",
        agent_type="dynamic",
        use_reflection=True,
        n_reflection=2,
        use_router=True
    )

if __name__ == "__main__":
    main()
