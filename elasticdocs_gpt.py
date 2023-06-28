import os
import streamlit as st
import openai
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# This code is part of an Elastic Blog showing how to combine
# Elasticsearch's search relevancy power with 
# OpenAI's GPT's Question Answering power
# https://www.elastic.co/blog/chatgpt-elasticsearch-openai-meets-private-data

# Code is presented for demo purposes but should not be used in production
# You may encounter exceptions which are not handled in the code


# Required Environment Variables
# openai_api - OpenAI API Key
# cloud_id - Elastic Cloud Deployment ID
# cloud_user - Elasticsearch Cluster User
# cloud_pass - Elasticsearch User Password

#openai.api_key = os.environ['openai_api']
model = "gpt-3.5-turbo-0301"

# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, http_auth=(user, passwd))
    return es


def check_env():
    if not load_dotenv(".env"):
        print("ERROR: load_dotenv returned error")
        return False

    env_list = ("cloud_id", "cloud_user", "cloud_pass", "openai_api_key")
    if all(env in os.environ for env in env_list):
        return True

    return False

def write_env(cid, cu, cp, oai_api):
    if not cid or not cu or not cp or not oai_api:
        return
    with open(".env", "w") as env_file:
        env_file.write("export cloud_id=" + cid + "\n")
        env_file.write("export cloud_user=" + cu + "\n")
        env_file.write("export cloud_pass=" + cp + "\n")
        env_file.write("export openai_api_key=" + oai_api + "\n")
        env_file.close()


# Search ElasticSearch index and return body and URL of the result
def search(query_text, cid, cu, cp, oai_api):
    if not cid and not cu and not cp and not oai_api:
        if check_env():
            print("reading from env file...")
            cid = os.environ['cloud_id']
            cp = os.environ['cloud_pass']
            cu = os.environ['cloud_user']
            openai.api_key = os.environ['openai_api_key']
        else:
            print("ERROR: Could not get environment variables!!!")
            return None, None
    else:
        print("writing to env file...")
        write_env(cid, cu, cp, oai_api)

###########################################################################
# 1. If cid, cu, cp, and oai_api is empty:
#   1 a. Check if env file exists
#       1 b. If file exists, read data file
#       1 c. If file doesn't exist, error out
#   1 d. If cid, cu, cp, and oai_api not empty:
#       1 e. Store data in .env file
###########################################################################
    es = es_connect(cid, cu, cp)

    # Elasticsearch query (BM25) and kNN configuration for hybrid search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "title": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }],
            "filter": [{
                "exists": {
                    "field": "title-vector"
                }
            }]
        }
    }

    knn = {
        "field": "title-vector",
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text
            }
        },
        "boost": 24
    }

    fields = ["title", "body_content", "url"]
    index = 'search-elastic-docs'
    resp = es.search(index=index,
                     query=query,
                     knn=knn,
                     fields=fields,
                     size=1,
                     source=False)

    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]

    return body, url

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

# Generate a response from ChatGPT based on the given prompt
def chat_gpt(prompt, model="gpt-3.5-turbo", max_tokens=1024, max_context_tokens=4000, safety_margin=5):
    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(prompt, max_context_tokens - max_tokens - safety_margin)

    response = openai.ChatCompletion.create(model=model,
                                            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": truncated_prompt}])

    return response["choices"][0]["message"]["content"]


st.title("ElasticDocs GPT")

# Main chat form
with st.form("chat_form"):
    cloud_id = st.text_input("cloud_id: ", type='password')
    username = st.text_input("username: ")
    password = st.text_input("password: ", type='password')
    oai_api = st.text_input("openai_api_key: ", type='password')
    query = st.text_input("You: ")
    submit_button = st.form_submit_button("Send")

# Generate and display response on form submission
negResponse = "I'm unable to answer the question based on the information I have from Elastic Docs."
if submit_button:
    resp, url = search(query, cloud_id, username, password, oai_api)
    prompt = f"Answer this question: {query}\nUsing only the information from this Elastic Doc: {resp}\nIf the answer is not contained in the supplied doc reply '{negResponse}' and nothing else"
    answer = chat_gpt(prompt)
    
    if negResponse in answer:
        st.write(f"ChatGPT: {answer.strip()}")
    else:
        st.write(f"ChatGPT: {answer.strip()}\n\nDocs: {url}")
