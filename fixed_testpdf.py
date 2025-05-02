from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
#define our LLM model 
from langchain_openai import ChatOpenAI
# from langchain_ollama import OllamaLLM
from langchain_community.chat_models import ChatOllama 


reader = PdfReader("C:\\Users\\vnsanthosh\\OneDrive - Virtusa\\Desktop\\Personal_Virtusa\\LLM_world\\sample_data\\2023_GPT4All_Technical_Report.pdf")
 
raw_text = ""

for i, page in enumerate(reader.pages): 
    text = page.extract_text() 
    if text: 
        raw_text += text


recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 50,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", "", "\t", "\r", "\n\r"]

)

rtexts = recursive_text_splitter.split_text(raw_text)

print(f"Total number of chunks: {len(rtexts)}")

rtexts_numbered=list(enumerate(rtexts, start=1))

print(f'the last chunk is: ', rtexts_numbered[-1])

for i, text in enumerate(rtexts):
    print(f"Chunk {i+1}::::::::::::", text )
    print(f"\n" )
    if i == 5:
        break

#use a HuggingFace model to generate embeddings for the text chunks

model_name = "sentence-transformers/all-MiniLM-L6-v2" 
embeddings = HuggingFaceEmbeddings(model_name=model_name)
# chunk_embeddings = embeddings.embed_documents(rtexts)

# create a FAISS embeddings and create index from the text chunks 
db = FAISS.from_texts(rtexts, embeddings)

# just for fun, let's see the embeddings and index saved in the disk

FAISS_INDEX_PATH = "my_test_faiss_index" # A folder name

print(f"\nSaving FAISS index to disk at: {FAISS_INDEX_PATH}")
db.save_local(folder_path=FAISS_INDEX_PATH) #glad there is a load_local method to load this index back :-), then i can reuse it and also can apped new index if i use same loaded db object. may be db.add_texts() method can be used to append new index.
print("Index saved.")


query = "who is narendra Modi? and is LLAMA a country?"

embeddings_search_result=db.similarity_search(query, k=10)
print(f"embeddings result: {embeddings_search_result}")
print(f"length of the embeddings result: {len(embeddings_search_result)}")

# now we can use the embeddings to generate a summary using a LLM

# tried with OpenAI API key but it is not working as i dont have quota :-) 

# model=ChatOpenAI(
#     openai_api_key=API_key,
#     model="gpt-3.5-turbo",
#     temperature=0.7,
#     max_tokens=1000,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
# )

# let me try with Ollama LLM running locally, dont need to speand a penny

model=ChatOllama(
    model="llama2",
    temperature=0.7,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# define our prompt template
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Prepare context for LLM
context = "\n".join([doc.page_content for doc in embeddings_search_result])
print(f"context: {context}")

template="""
You are a helpful assistant that can answer any questions based on below text only. If anything outside the text is asked, say "I don't know".
{text}
question: {question}"""


prompt=ChatPromptTemplate.from_template(template)

prompt_value = prompt.format_prompt(text=context, question=query)



response = model.generate(
    messages=[prompt_value.to_messages()]
)

print("Response:", response.generations[0][0].text)




























