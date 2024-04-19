import docx
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import pinecone
import openai 

# Initialize Pinecone client
pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='us-west1')
index = pinecone.Index(name='vector_database')

# openai API Key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Loading the text from the provided docx file and split into chunks
def load_and_split_text(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    # Split the text into chunks
    chunks = text.split("\n\n")  # Adjust the delimiter as needed
    return chunks

# Function to add data to the vector database
def add_data_to_database(chunks):
    # Load the embedding model
    tokenizer = AutoTokenizer.from_pretrained("text-embedding-ada-002")
    model = AutoModel.from_pretrained("text-embedding-ada-002")

    # Process each chunk and add to database
    for chunk in chunks:
        # Tokenize and encode the chunk
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        # Pass the inputs through the model to get the embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract the embeddings (CLS token)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        # Add the chunk and its embedding to the database
        index.upsert(chunks=[chunk], vectors=embeddings)

# Function to create prompt for GPT-3
def create_prompt(user_query, best_matches):
    # Combine user query with best matches from vector database
    prompt = "User Query: " + user_query + "\n"
    prompt += "Contexts:\n"
    for match in best_matches:
        prompt += match + "\n"
    return prompt

# Function to get answer from GPT-3
def get_answer_from_gpt3(prompt):
    # Call the OpenAI API to get the response
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Function to find best matches from vector database
def find_best_matches(query):
    # Tokenize and encode the query
    tokenizer = AutoTokenizer.from_pretrained("text-embedding-ada-002")
    model = AutoModel.from_pretrained("text-embedding-ada-002")
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
    # Perform nearest neighbor search in Pinecone index
    results = index.query(queries=[query_embedding], top_k=3)  # Adjust top_k as needed
    best_matches = [result['id'] for result in results[0]]
    return best_matches

# Function to handle user query and get a solid answer
def user_query(query):
    # Load and split text from docx file
    chunks = load_and_split_text("DataLaw.docx")
    # Add splitted chunks to the vector database
    add_data_to_database(chunks)
    # Find best matches for user query from vector database
    best_matches = find_best_matches(query)
    # Create prompt for GPT-3
    prompt = create_prompt(query, best_matches)
    # Get answer from GPT-3
    answer = get_answer_from_gpt3(prompt)
    return answer

# Main function
def main():
    # Sample user query
    query = "How does climate change affect biodiversity?"
    # Handle user query and get solid answer
    answer = user_query(query)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
