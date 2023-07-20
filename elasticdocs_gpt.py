import os
import streamlit as st
import openai
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import tiktoken
import time

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
    es = Elasticsearch(cloud_id=cid, basic_auth=(user, passwd))
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

    es = es_connect(cid, cu, cp)

    if "query_field" in os.environ:
        query_field = os.environ["query_field"]
    else:
        query_field = "title-vector"
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
                    "field": query_field
                }
            }]
        }
    }

    knn = {
        "field": query_field,
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
    if "es_index" in os.environ:
        index = os.environ["es_index"]
    else:
        index = 'search-elastic-docs'
    print(f"using index {index}")
    search_results["vector"] = es.search(index=index,
                     query=query,
                     knn=knn,
                     fields=fields,
                     size=1,
                     source=False)
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


def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text, len(tokens)

    return ' '.join(tokens[:max_tokens]), len(tokens)


def encoding_token_count(string: str, encoding_model: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_model)
    return len(encoding.encode(string))


# Generate a response from ChatGPT based on the given prompt
def chat_gpt(prompt, model="gpt-3.5-turbo", max_tokens=1024, max_context_tokens=4000, safety_margin=5):
    # Truncate the prompt content to fit within the model's context length
    truncated_prompt, word_count = truncate_text(prompt, max_context_tokens - max_tokens - safety_margin)
    openai_token_count = encoding_token_count(prompt, model)
    print(f"word_count = {word_count}, openai_token_count = {openai_token_count}")
    response = openai.ChatCompletion.create(model=model,
                                            messages=[
                                                {"role": "system", "content": "You are a helpful assistant."},
                                                {"role": "user", "content": truncated_prompt}
                                                      ])

    return response["choices"][0]["message"]["content"], word_count, response["usage"]["total_tokens"]


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
        search(query, cloud_id, username, password, oai_api)
        # Setup columns for different search results
        s_col = {}
        s_col["bm25"], s_col["vector"],s_col["elser"] = st.columns(3)
        s_col["bm25"].write("# BM25")
        s_col["vector"].write("# Basic Vector")
        s_col["elser"].write("# Elser")

        for s in search_results.keys():
            col = s_col[s]
            try:
                body = search_results[s]['hits']['hits'][0]['fields']['body_content'][0]
                url = search_results[s]['hits']['hits'][0]['fields']['url'][0]
                prompt = f"Answer this question: {query}\nUsing only the information from this Elastic Doc: {body}\nIf the answer is not contained in the supplied doc reply '{negResponse}' and nothing else"
                begin = time.perf_counter()
                answer, word_count, openai_token_count = chat_gpt(prompt)
                end = time.perf_counter()
                answer_token_count = encoding_token_count(answer, "gpt-3.5-turbo")
                cost = float((0.0015*(openai_token_count)/1000) + (0.002*(answer_token_count/1000)))
                time_taken = end - begin
                col.write("## ChatGPT Response")
                if negResponse in answer:
                    col.write(f"\n\n**Word count: {word_count}, Token count: {openai_token_count}**")
                    col.write(f"\n**Cost: ${cost:0.6f}, ChatGPT response time: {time_taken:0.4f} sec**")
                    col.write(f"{answer.strip()}")
                else:
                    col.write(f"\n\n**Word count: {word_count}, Token count: {openai_token_count}**")
                    col.write(f"\n**Cost: ${cost:0.6f}, ChatGPT response time: {time_taken:0.4f} sec**")
                    col.write(f"{answer.strip()}\n\nDocs: {url}")
                col.write("---")
                col.write(f"## Elasticsearch {s} response:")
                try:
                    col.write(search_results[s]['hits']['hits'][0]['fields'])
                except:
                    col.write("No results yet!")
            except IndexError as e:
                col.write("### No search results returned")
                



if __name__ == "__main__":
    main()
