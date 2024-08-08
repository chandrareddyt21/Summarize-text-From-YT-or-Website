import sys, os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit app
st.set_page_config(page_title="Langchain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 Langchain: Chat with SQL DB")
st.subheader("Summarize URL")

# Get the GROQ API KEY and url(YT or Website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API key:", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# loading model
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """"
Provide a summary of the following content in 300 words
Content:{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['text'])


if st.button("Summarize the content from YT or Website"):
    # Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please fill all the fields")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can may be a YT video or Website url")
    else:
        try:
            with st.spinner("Waiting..."):
                # Loading the data from provide url
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], 
                        ssl_verify=False, 
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                
                docs = loader.load()

                # Chain for summarization
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.invoke(docs)

                st.success(output_summary['output_text'])
        except Exception as e:
            st.exception(e)