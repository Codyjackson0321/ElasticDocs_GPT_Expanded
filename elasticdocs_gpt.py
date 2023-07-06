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
        print("ENV file found!")
        return True

    return False

def write_env(cid, cu, cp, oai_api):
    if not cid or not cu or not cp or not oai_api:
        return
    with open(".env", "w") as env_file:
        env_file.write("cloud_id=\"" + cid + "\"\n")
        env_file.write("cloud_user=\"" + cu + "\"\n")
        env_file.write("cloud_pass=\"" + cp + "\"\n")
        env_file.write("openai_api_key=\"" + oai_api + "\"\n")
        env_file.close()


search_results = {
    "elser": {},
    "vector": {},
    "bm25": {},
}
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
        openai.api_key = oai_api

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
    index = 'search-elastic-docs-completed'
    resp = es.search(index=index,
                     query=query,
                     knn=knn,
                     fields=fields,
                     size=1,
                     source=False)
    #print(resp)



    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]



    # Begin setting additional search results:
    # Vector
    search_results['vector'] = resp

    # Generic BM25
    search_results['bm25'] = es.search(
        index=index,
        query=query,
        fields=fields,
        size=1,
        source=False
    )

    # Elser
    search_results['elser'] = es.search(
        index=index,
        query={
            "bool": {
                "should": [
                    {
                        "text_expansion": {
                            "ml.inference.body_content_expanded.predicted_value": {
                                "model_id": ".elser_model_1",
                                "model_text": query_text
                            }
                        }
                    },
                    {
                        "text_expansion": {
                            "ml.inference.title_expanded.predicted_value": {
                                "model_id": ".elser_model_1",
                                "model_text": query_text
                            }
                        }
                    }
                ]
            }
        },
        fields=fields,
        size=1,
        source=False
    )

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
                                            messages=[
                                                {"role": "system", "content": "You are a helpful assistant."},
                                                {"role": "user", "content": truncated_prompt}
                                                      ])

    return response["choices"][0]["message"]["content"]


def main():
    st.set_page_config(
        layout="wide"
    )
    st.title("ElasticDocs GPT")

    # Main chat form
    with st.form("chat_form"):
        input_col1, input_col2, input_col3, input_col4 = st.columns(4)
        cloud_id = input_col1.text_input("cloud_id: ", type='password')
        username = input_col2.text_input("username: ")
        password = input_col3.text_input("password: ", type='password')
        oai_api = input_col4.text_input("openai_api_key: ", type='password')
        query = st.text_input("You: ")
        submit_button = st.form_submit_button("Send")

    # Generate and display response on form submission
    negResponse = "I'm unable to answer the question based on the information I have from Elastic Docs."
    if submit_button:
        resp, url = search(query, cloud_id, username, password, oai_api)
        print(resp)
        prompt = f"Answer this question: {query}\nUsing only the information from this Elastic Doc: {resp}\nIf the answer is not contained in the supplied doc reply '{negResponse}' and nothing else"
        answer = chat_gpt(prompt)

        # Setup columns for different search results
        gpt_col, bm25_col, vector_col, elser_col = st.columns(4)
        gpt_col.header("ChatGPT")
        bm25_col.header("BM25")
        vector_col.header("Basic Vector")
        elser_col.header("Elser")

        # Sets ChatGPT answer
        if negResponse in answer:
            gpt_col.write(f"ChatGPT: {answer.strip()}")
        else:
            gpt_col.write(f"ChatGPT: {answer.strip()}\n\nDocs: {url}")

        # Sets BM25 response
        try:
            bm25_col.write(search_results['bm25']['hits']['hits'][0]['fields'])
        except:
            bm25_col.write("No results yet!")

        # Sets BM25 response
        try:
            vector_col.write(search_results['vector']['hits']['hits'][0]['fields'])
        except:
            vector_col.write("No results yet!")


        try:
            elser_col.write(search_results['elser']['hits']['hits'][0]['fields'])
        except:
            elser_col.write("No results yet!")

if __name__ == "__main__":
    main()
