from langchain_groq import ChatGroq
import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import uuid

llm = ChatGroq(
    temperature=0,
    api_key = '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',
    model="llama-3.1-70b-versatile"
)

page_data = loader.load().pop().page_content
# print(pagloader = WebBaseLoader("https://www.google.com/about/careers/applications/jobs/results/110690555461018310-software-engineer-iii-infrastructure-core")e_data)

prompt_extract = PromptTemplate.from_template(
    """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """
)

chain_extract = prompt_extract | llm
res = chain_extract.invoke(input={'page_data': page_data})
# print(res.content)

json_parser = JsonOutputParser()
json_response = json_parser.parse(res.content)
# print(json_response)
# type(json_response)
df = pd.read_csv("my_portfolio.csv")
# print(df)

client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

if not collection.count():
    for _, row in df.iterrows():
        collection.add(documents=row["Techstack"],
                       metadatas={"links": row["Links"]},
                       ids=[str(uuid.uuid4())])
        
links = collection.query(query_texts=['skills'], n_results=2).get('metadatas', [])
# print(links)

job = json_response
# print(job['skills'])
prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Parth Sharma, a 2nd year Student in pursuing Btech.  
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of XYZ 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase XYZ's portfolio: {link_list}
        Remember you are Parth Sharma, A student of Btech 2nd Year. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        """
        )

chain_email = prompt_email | llm
res = chain_email.invoke({"job_description": str(job), "link_list": links})
print(res.content)