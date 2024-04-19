Text Processing with GPT-3

This project demonstrates how to process user queries, generate prompts for GPT-3, and obtain responses using Python and the OpenAI GPT-3 API. The project also integrates with Pinecone for storing and retrieving vector embeddings of text data.

Features:

- Load text from a .docx file and split it into chunks.
- Add text chunks to a vector database using Pinecone.
- Generate prompts for GPT-3 based on user queries and context from the vector database.
- Obtain responses from GPT-3 based on the generated prompts.
- Change the embedding model to OpenAI's 'text-embedding-ada-002'.
- Suggestions for improvement and an alternative approach summary.
  


Usage:
Modify the script.py file to customize the behavior of the application.
Replace placeholders with actual API keys, file paths, and other relevant information.
Run the script to process user queries and obtain responses from GPT-3.

Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.

License
This project is licensed under the MIT License
