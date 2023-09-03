import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from dotenv import load_dotenv
from supabase.client import Client, create_client

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Set the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Directory containing the PDFs
directory = "whitepapers"

# Initialize the text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
vector_store = SupabaseVectorStore(embeddings, client=supabase, table_name='whitepapers')

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(directory, filename)
        
        # Load and split the document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        
        # Add the document to the vector store
        vector_store.add_documents(docs)

#query = "Whats the recommended course of action to fix the artic crisis?"
#matched_docs = vector_store.similarity_search(query)
#print(matched_docs[0].page_content)