import os
import streamlit as st
import openai
from elasticsearch import Elasticsearch
from dotenv import dotenv_values
import tiktoken
import time
import ipaddress
import re

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

models = {
    "gpt-3.5-4k-tokens": {
        "name": "gpt-3.5-turbo", 
        "token_length": 4096,
        "input_cost": 0.0015,
        "output_cost": 0.002,
        "available": True 
    },
    "gpt-3.5-16k-tokens": {
        "name": "gpt-3.5-turbo-16k", 
        "token_length": 16384,
        "input_cost": 0.003,
        "output_cost": 0.004,
        "available": True 
    },
    "gpt-4-8k-tokens": {
        "name": "gpt-4", 
        "token_length": 8192,
        "input_cost": 0.03,
        "output_cost": 0.06,
        "available": False 
    },
    "gpt-4-32k-tokens": {
        "name": "gpt-4-32k", 
        "token_length": 32768,
        "input_cost": 0.06,
        "output_cost": 0.12,
        "available": False 
    }
}

# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    if is_valid_cloud_id(cid):
        # ESS cluster use the normal way
        es = Elasticsearch(cloud_id=cid, basic_auth=(user, passwd))
    elif is_valid_ip(cid):
        # IP provided, we have to insert https & port and assume it's valid cert
        url=f"https://{cid}:9200"
        es = Elasticsearch(hosts=[url], basic_auth=(user, passwd))
    elif is_valid_url(cid):
        if 'localhost' in cid:
            if 'ca_certs' in os.environ:
                es = Elasticsearch(hosts=[cid], basic_auth=(user, passwd), ca_certs=os.environ['ca_certs'])
            else:
                es = Elasticsearch(hosts=[cid], basic_auth=(user, passwd), verify_certs=False)
        else:
            es = Elasticsearch(url, basic_auth=(user, passwd))
    elif 'localhost' in cid:
        url = f"https://{cid}"
        if 'ca_certs' in os.environ:
            es = Elasticsearch(hosts=[url], basic_auth=(user, passwd), ca_certs=os.environ['ca_certs'])
        else:
            es = Elasticsearch(url, basic_auth=(user, passwd), verify_certs=False)
    else:
        print(f"ERROR: Invalid cid = {cid}")
        return None

    return es


def is_valid_ip(ip: str) -> bool:
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def is_valid_url(url: str) -> bool:
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return bool(url_pattern.match(url))


def is_valid_cloud_id(cloud_id: str) -> bool:
    import base64
    cloud_id_pattern = re.compile(r'^[a-zA-Z0-9_-]+:[a-zA-Z0-9+/=]+$')
    if not cloud_id_pattern.match(cloud_id):
        return False
    cluster_name, data = cloud_id.split(':')
    try:
        decoded_data = base64.b64decode(data, validate=True).decode()
        cloud_fqdn = decoded_data.split(':')[0]
        print(f"cluster: {cluster_name}.{cloud_fqdn}")
        return True
    except (base64.binascii.Error, TypeError):
        print(f"cloud_id --> {cloud_id} doesn't have base64")
        print(f"decoded_data {decoded_data}")
        return False


env_list = ("cloud_id", "cloud_user", "cloud_pass", "openai_api_key")
def check_env(**kwargs):
    if not (env_file_config := dotenv_values(".env")):
        print("couldn't read .env file")
        env = {key: kwargs.get(key, None) for key in env_list}
    else:
        env = {key: kwargs.get(key) if kwargs.get(key) not in ("", None) else env_file_config.get(key, None) for key in set(env_list) | set(env_file_config)}

    if (no_env_list := [e for e in env_list if e not in env or env[e] == '']):
        print(f"ERROR: the following are not in the list {no_env_list}")
        return False

    with open(".env", "w") as env_file:
        for k in [key for key, value in env.items() if value is not None]:
            env_file.write(f"{k}=\"{env[k]}\"\n")
            os.environ[k] = env[k]
    env_file.close()
    return True


search_results = {
    "elser": {},
    "vector": {},
    "bm25": {},
}
# Search ElasticSearch index and return body and URL of the result
def search(query_text, index="search-elastic-docs"):
    cid = os.environ['cloud_id']
    cp = os.environ['cloud_pass']
    cu = os.environ['cloud_user']
    openai.api_key = os.environ['openai_api_key']

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
def chat_gpt(prompt, model, max_tokens=1024):
    # Truncate the prompt content to fit within the model's context length
    safety_margin = int(models[model]['token_length']*0.25)
    truncated_prompt, word_count = truncate_text(prompt, models[model]['token_length'] - max_tokens - safety_margin)
    openai_token_count = encoding_token_count(prompt, models[model]['name'])
    print(f"word_count = {word_count}, openai_token_count = {openai_token_count}")
    response = openai.ChatCompletion.create(model=models[model]['name'],
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
        cloud_id = input_col1.text_input("cloud_id/host: ", type='password')
        username = input_col2.text_input("username: ")
        password = input_col3.text_input("password: ", type='password')
        oai_api = input_col4.text_input("openai_api_key: ", type='password')
        model_option = st.selectbox(
            'Choose LLM Model',
            [key for key, val in models.items() if val["available"]]
        )
        index_option = st.selectbox(
            'Choose Elasticsearch Index',
            ['search-elastic-docs-completed', 'search-elastic-docs'],
            index=1
        )
        query = st.text_input("Enter your query: ")
        submit_button = st.form_submit_button("Send")

    # Generate and display response on form submission
    negResponse = "I'm unable to answer the question based on the information I have from Elastic Docs."
    if submit_button:
        print(f"selected model {model_option}")
        if not check_env(cloud_id=cloud_id, cloud_user=username, cloud_pass=password, openai_api_key=oai_api):
            st.write("ERROR environment variables not set!!!")
        else:
            search(query, index=index_option)
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
                    answer, word_count, openai_token_count = chat_gpt(prompt, model=model_option)
                    end = time.perf_counter()
                    answer_token_count = encoding_token_count(answer, models[model_option]['name'])
                    input_model_cost = models[model_option]["input_cost"]
                    output_model_cost = models[model_option]["output_cost"]
                    cost = float((input_model_cost*(openai_token_count)/1000) + (output_model_cost*(answer_token_count/1000)))
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
