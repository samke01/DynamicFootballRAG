import json
import faiss
import numpy as np
import pandas as pd
import os
import nest_asyncio
import asyncio
import aiohttp
import tiktoken
import time
import logging
import gc

from sklearn.preprocessing import normalize
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# Apply asyncio nest
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_logs.txt"),  # Log to a file
    ]
)

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Set it as an environment variable 'OPENAI_API_KEY'.")

class RAGPipeline:
    def __init__(self, openai_embedding_model="text-embedding-3-large", reranking_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', openai_llm= 'gpt-4o-mini', embedding_token_limit=8192):
        """Initialize the RAG pipeline with an embedding model, a reranking model, and Ollama LLM for query expansion."""
        self.embedding_model_name = openai_embedding_model  # Store model name as a string
        self.embedding_model = OpenAIEmbeddings(model=openai_embedding_model)
        self.reranking_model = CrossEncoder(reranking_model_name, device='cuda')
        self.llm = ChatOpenAI(model=openai_llm, temperature=0.0)
        self.token_limit = embedding_token_limit  # Maximum tokens per embedding request
        self.vectorstore = None
        self.data = pd.DataFrame(columns=['content', 'metadata', 'embedding'])
        self.embedding_dim = len(self.embedding_model.embed_documents(["test"])[0])  # Cache the embedding size
        self.team_mapping = {1730: "Augsburg", 134: "Borussia M.Gladbach", 44: "Borussia Dortmund", 282: "FC Koln", 1147: "Darmstadt", 45: "Eintracht Frankfurt", 1211: "Hoffenheim", 50: "Freiburg", 36: "Leverkusen", 7614: "RBL", 109: "Bochum", 41: "Stuttgart", 796: "Union Berlin", 219: "Mainz", 42: "Werder Bremen", 37: "Bayern", 4852: "FC Heidenheim", 33: "Wolfsburg"}
        
    @staticmethod
    def create_token_safe_batches(chunks, encoding, max_tokens):
        """
        Create batches of chunks that stay within the token limit.
        
        Args:
            chunks (list): List of text chunks.
            encoding (tiktoken.Encoding): Tokenizer for the model.
            max_tokens (int): Maximum tokens allowed per batch.
            
        Returns:
            list: List of token-safe batches.
        """
        batches = []
        current_batch = []
        current_token_count = 0
        
        for chunk in chunks:
            token_count = len(encoding.encode(chunk))  # Count tokens in the current chunk
            
            # If adding this chunk exceeds the token limit, start a new batch
            if current_token_count + token_count > max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_token_count = 0
                
            # Add the chunk to the current batch
            current_batch.append(chunk)
            current_token_count += token_count
            
        # Add the last batch if it has content
        if current_batch:
            batches.append(current_batch)
            
        return batches

    @retry(
        stop=stop_after_attempt(15),  # Retry up to 15 times
        wait=wait_exponential(multiplier=2, min=15, max=60),  # Exponential backoff
        before_sleep=before_sleep_log(logger, logging.INFO)  # Log before retrying
    )
    async def _fetch_embedding(self, session, batch, url, headers):
        """
        Fetch embeddings with improved error handling for quota exhaustion.
        """
        async with session.post(
            url,
            json={"model": self.embedding_model_name, "input": batch},
            headers=headers,
        ) as response:
            if response.status == 429:  # Rate limit hit
                retry_after = int(response.headers.get("x-ratelimit-reset-tokens", 60))
                logger.info(f"Rate limit hit. Retrying in {retry_after} seconds...")
                raise Exception("Rate limit hit")
            elif response.status == 400 and "insufficient_quota" in await response.text():
                logger.error("Quota exhausted. Upgrade your plan or increase your limits.")
                raise Exception("Quota exhausted")
            elif response.status != 200:
                logger.error(f"API Error: {response.status}, {await response.text()}")
                raise Exception("API Error")
            result = await response.json()
            return [item['embedding'] for item in result.get('data', [])]

    async def _generate_embeddings_async(self, batches, max_concurrent_requests=10):
        """
        Generate embeddings for token-safe batches using parallel API calls with rate limit handling.
        """
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }
        
        embeddings = []
        
        # Limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        async with aiohttp.ClientSession() as session:
            async def process_batch(batch, batch_idx):
                async with semaphore:
                    logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}...")
                    batch_embeddings = await self._fetch_embedding(session, batch, url, headers)
                    return batch_embeddings
                
            # Process all batches
            tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
            results = await asyncio.gather(*tasks)
            
        # Flatten the results
        embeddings = [embedding for batch_result in results for embedding in batch_result]
        logger.info(f"Generated embeddings for {len(embeddings)} inputs.")
        return embeddings

    def generate_batch_embeddings(self, chunks, save_embeddings=True):
        """
        Generate embeddings using asynchronous parallel API calls with dynamic batch sizing.
        """
        # Validate and preprocess chunks
        encoding = tiktoken.encoding_for_model(self.embedding_model_name)
        max_tokens_per_input = 8191
        
        # Validate and preprocess chunks
        for i, chunk in enumerate(chunks):
            if 'content' not in chunk or not isinstance(chunk['content'], str):
                print(f"Adding placeholder content for missing content in chunk at index {i}.")
                
                if chunk.get('metadata', {}).get('entity') == "LEAGUE":
                    
                    not_played = (
                        f"Bundesliga Season 2023/2024 │ Game Week: {chunk['metadata']['gameweek']}\n\n"
                        
                        f"The team {self.team_mapping[chunk['metadata']['team_id']]} did not play during game week {chunk['metadata']['gameweek']}."
                        f"As a result, the team has one fewer match played than the rest of the competition, "
                        f"which should be considered when analyzing the league standings as of this game week.\n\n"
                        )
                    
                    chunk['content'] = not_played
                else:
                    chunk['content'] = "Placeholder content. Original content is missing."
                    
            if 'metadata' not in chunk or not isinstance(chunk['metadata'], dict):
                chunk['metadata'] = {"info": "Missing metadata"}
                
            token_count = len(encoding.encode(chunk['content']))
            if token_count > max_tokens_per_input:
                raise ValueError(f"Chunk at index {i} exceeds the token limit ({max_tokens_per_input}).")
            
        # Extract content and metadata
        contents = [chunk['content'] for chunk in chunks]
        metadata = [chunk['metadata'] for chunk in chunks]
        
        # Create token-safe batches
        batches = self.create_token_safe_batches(contents, encoding, max_tokens=self.token_limit)
        logger.info(f"Created {len(batches)} token-safe batches for processing.")
        
        # Determine maximum concurrent requests based on RPM
        max_concurrent_requests = min(3000 // 60, len(batches))  # Respect RPM limit
        embeddings = asyncio.run(self._generate_embeddings_async(batches, max_concurrent_requests))
        
        # Ensure lists are of equal length
        if len(contents) != len(metadata) or len(contents) != len(embeddings):
            raise ValueError("Length mismatch: contents, metadata, and embeddings must have the same length.")
        
        # Save results
        self.data = pd.DataFrame({'content': contents, 'metadata': metadata, 'embedding': embeddings})
        if save_embeddings:
            self.save_embeddings("embeddings.json")
            logger.info("Embeddings saved to 'embeddings.json'.")
        return embeddings
    
    
    def save_embeddings_to_faiss(self, index_name="faiss_index"):
        """
        Save normalized embeddings, chunks, and IDs into a FAISS index for cosine similarity.

        Args:
            index_name (str): Name of the FAISS index to save.
        """
        if self.data.empty:
            raise ValueError("The data object is empty. Generate embeddings before saving to FAISS.")

        # Step 1: Extract embeddings, content, and metadata
        embeddings = np.vstack(self.data['embedding'].to_list())
        contents = self.data['content'].tolist()
        metadata = self.data['metadata'].tolist()

        # Step 2: Normalize embeddings for cosine similarity
        embeddings = normalize(embeddings, norm='l2', axis=1)

        # Step 3: Initialize FAISS index and add normalized embeddings
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)

        # Step 4: Prepare LangChain metadata and docstore
        documents = [Document(page_content=content, metadata=meta) for content, meta in zip(contents, metadata)]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        # Step 5: Wrap FAISS index with LangChain
        self.vectorstore = FAISS(
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=self.embedding_model
        )

        # Step 6: Save the index
        self.vectorstore.save_local(index_name)
        print(f"FAISS index saved as '{index_name}' with cosine similarity enabled.")

    def create_faiss_index(self, chunks, index_name="faiss_index", save_embeddings=True):
        """
        Create a FAISS index from the given text chunks.

        Args:
            chunks (dict): dictionary that contains 'content' and 'metadata' keys.
            index_name (str): Name of the FAISS index to save.
            batch_size (int): Number of chunks to process per batch when generating the embeddings.
        Returns:
            None
        """
        self.generate_batch_embeddings(chunks, save_embeddings=save_embeddings)
        self.save_embeddings_to_faiss(index_name=index_name)

    def load_faiss_index(self, index_name="faiss_index"):
        """Load a FAISS index from a file."""
        self.vectorstore = FAISS.load_local(index_name, self.embedding_model, allow_dangerous_deserialization=True)

    def save_faiss_index(self, filepath):
        """Save the FAISS index to a file."""
        if self.vectorstore:
            self.vectorstore.save_local(filepath)
        else:
            print("FAISS index is not initialized.")
    
    def save_embeddings(self, filepath):
        """Save the embeddings and metadata to a JSON file."""
        self.data.to_json(filepath, orient='records', lines=True)

    def load_embeddings(self, filepath):
        """Load embeddings and metadata from a JSON file."""
        self.data = pd.read_json(filepath, orient='records', lines=True)
        self.data['embedding'] = self.data['embedding'].apply(np.array)

    def load_json(self, filepath):
        """Load a JSON file containing chunks in the required format."""
        with open(filepath, 'r') as file:
            chunks = json.load(file)
        return chunks
    
    def load_dataframe_from_json(self, filepath):
        """Load a JSON file containing a DataFrame."""
        return pd.read_json(filepath, orient='records', lines=True)
    
    def expand_query(self, query):
        """
        Expand the query using the LLM to generate alternative versions.

        Args:
            query (str): Original user query.

        Returns:
            list: List of expanded queries.
        """
        # Define the system prompt
        system_prompt = SystemMessagePromptTemplate.from_template ("""You are an expert at converting user questions into RAG queries. 
        You have access to a RAG framework based on football (soccer) events that describe actions during a match. The RAG framework is particularly bad when it comes to retrieving number related queries numbers (e.g. "In what minute?", "In what field zone?", "From which to which field zone?"). 

        Perform query expansion. If there are multiple common ways of phrasing a user question 
        or common synonyms for key words in the question, make sure to return multiple versions 
        of the query with the different phrasings. 

        If there are acronyms or words you are not familiar with, do not try to rephrase them.

        Return 5 versions of the question separated by newlines. Only provide the query, no numbering."""
        )

        human_prompt = HumanMessagePromptTemplate.from_template("{question}")
        chat_prompt = ChatPromptTemplate(messages=[system_prompt, human_prompt])

        # Generate the expanded queries
        prompt = chat_prompt.format_prompt(question=query)
        response = self.llm(prompt.to_messages())
        
        # Parse and return the expanded queries
        expanded_queries = [
            q.strip() for q in response.content.split('\n') if q.strip()
        ]
        if query not in expanded_queries:
            expanded_queries.insert(0, query)  # Ensure original query is included
            
        return expanded_queries
    
    def retrieve(self, queries, top_k=50):
        """
        Retrieve the top-k similar chunks for each expanded query from FAISS.

        Args:
            queries (list): List of expanded queries.
            top_k (int): Number of results to retrieve per query.

        Returns:
            list: Combined results from all queries.
        """ 
        # if queries is a single string, convert it to a list
        if type(queries) != list:
            try:
                queries = [queries]
            except:
                raise ValueError("Queries must be a list of strings or in case of one query a single string.")

        results = []    
        for query in queries:
            retrieved_docs = self.vectorstore.similarity_search_with_score(query, k=top_k)
            results.extend(retrieved_docs)
            
        # extract the score to the metadata removing the tuple
        for i, doc in enumerate(results):
            doc[0].metadata["similarity_score"] = doc[1]
            results[i] = doc[0]

        # Remove duplicates based on the document content
        unique_results = list({doc.page_content: doc for doc in results}.values())
        
        return unique_results
    
    def reorder(self, chunks):
        """
        Reorder chunks to mitigate the 'Lost in the Middle' phenomenon.

        Args:
            chunks (list): List of chunks sorted by relevance score in descending order.

        Returns:
            list: Reordered chunks.
        """
        n = len(chunks)
        reordered = [None] * n  # Initialize a list to store reordered chunks
        left = 0
        right = n - 1

        for i in range(n):
            if i % 2 == 0:
                # Even step: Pick from the left side
                reordered[i] = chunks[left]
                left += 1
            else:
                # Odd step: Pick from the right side
                reordered[i] = chunks[right]
                right -= 1

        return reordered

    def rerank(self, query, retrieved_chunks):
        """
        Re-rank the retrieved chunks using the CrossEncoder's rank method, addressing the 'Lost in the Middle' phenomenon.
        
        Args:
            query (str): The user query.
            retrieved_chunks (list): List of retrieved documents to be re-ranked.
            
        Returns:
            list: Re-ranked documents, reordered to mitigate 'Lost in the Middle' effects.
        """
        # Extract document contents
        documents = [chunk.page_content for chunk in retrieved_chunks]
        
        # Use CrossEncoder's rank method to score documents
        ranked_results = self.reranking_model.rank(query, documents, return_documents=True)
        
        # Update metadata with scores
        for chunk, ranked_doc in zip(retrieved_chunks, ranked_results):
            chunk.metadata['rerank_score'] = ranked_doc['score']
        
        return retrieved_chunks
    
    def answer_query(
        self, 
        query, 
        kwargs=
            {
                "retrieve": 100, 
                "top_k": 100, 
                "threshold": 0.2, 
                "reorder": True,
                "alpha": 1.0,
                "beta": 1.0
            }
        ):
        """Answer the query using the LLM and the top-k re-ranked documents.
        Args:
            query (str): The user query.
            kwargs (dict): Keyword arguments for the retrieval and re-ranking process.
                - retrieve (int): Number of documents to retrieve from the FAISS index.
                - top_k (int): Number of documents to return.
                - threshold (float): Minimum similarity score to consider for retrieval.
                - reorder (bool): Whether to reorder the top-k documents to mitigate 'Lost in the Middle'.
                - alpha (float): Weight for FAISS (semantic similarity) scores.
                - beta (float): Weight for BM25 (lexical similarity) scores.
        """
        
        # Expand the query to generate alternative versions
        expanded_queries = self.expand_query(query)
        
        # Retrieve the top-k similar chunks for each expanded query
        retrieved_chunks = self.retrieve(expanded_queries, top_k=kwargs["retrieve"])
        
        # Re-rank the retrieved chunks using the CrossEncoder
        ranked_chunks = self.rerank(query, retrieved_chunks)
        
        # Sort chunks by their RSE values in descending order
        ranked_chunks.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        
        # Reorder the top-k ranked chunks to mitigate 'Lost in the Middle'
        if kwargs["reorder"]:
            reordered_chunks = self.reorder(ranked_chunks[:kwargs["top_k"]])
            final_chunks = reordered_chunks
        else:
            final_chunks = ranked_chunks[:kwargs["top_k"]]
            
        # Rerank scores
        rerank_scores = [chunk.metadata["rerank_score"] for chunk in final_chunks]
        
        # Similarity scores
        similarity_scores = [chunk.metadata.get("similarity_score", 0) for chunk in final_chunks]
        # Combine the top-k ranked chunks for the LLM
        context = "\n\n".join([chunk.page_content for chunk in final_chunks])

        # Define the structure for the prompt
        system_prompt = SystemMessagePromptTemplate.from_template(
            """ You are an expert at answering a user query based on a provided context. Analyze the context with deep reasoning and provide a concise and informative answer to the user query.
                Keep your answer ground in the facts of the Context. Always provide a clear, concise and short answer that directly answers the users query. Do not provide additional information irrelevant to the user query. Do not provide your reasoning or thought process. Do not format the reply in any way. Only provide the answer to the user query.
            """

        )
        human_prompt = HumanMessagePromptTemplate.from_template(
            """
            User Query:
            {query}
            
            Context:
            {context}
            """
        )
        chat_prompt = ChatPromptTemplate(messages=[system_prompt, human_prompt])

        # Format the prompt with the query and context
        prompt = chat_prompt.format_prompt(query=query, context=context)
        response = self.llm(prompt.to_messages())
        return response.content, context, rerank_scores, similarity_scores, expanded_queries

class BatchProcessor:
    def __init__(self, model, batch_size=10000, output_dir="embedding_batches"):
        """
        Initialize the batch processor.

        Args:
            model (RAGPipeline): The pipeline for embedding generation.
            batch_size (int): Number of entries to process in each batch.
            output_dir (str): Directory to save the embedding results.
        """
        self.model = model
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def split_into_batches(data, batch_size):
        """
        Split dataset into batches.

        Args:
            data (list): List of entries, where each entry has 'content' and 'metadata'.
            batch_size (int): Number of entries per batch.

        Returns:
            generator: Yields batches of data.
        """
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def process_batch(self, batch, batch_idx):
        """
        Process a single batch to generate embeddings and save results.

        Args:
            batch (list): List of entries in the batch.
            batch_idx (int): Index of the batch being processed.

        Returns:
            None
        """
        print(f"Processing batch {batch_idx + 1}...")

        # Generate embeddings
        embeddings = self.model.generate_batch_embeddings(batch, save_embeddings=False)

        # Save embeddings to a file
        output_file = os.path.join(self.output_dir, f"embeddings_batch_{batch_idx + 1}.json")
        with open(output_file, "w") as f:
            json.dump(embeddings, f)

        print(f"Saved batch {batch_idx + 1} to {output_file}")

        # Clear memory
        del embeddings
        del batch
        gc.collect()

    def process_all_batches(self, data):
        """
        Process all batches in the dataset.

        Args:
            data (list): List of entries to process.

        Returns:
            None
        """
        for batch_idx, batch in enumerate(self.split_into_batches(data, self.batch_size)):
            self.process_batch(batch, batch_idx)

    @staticmethod
    def merge_batches(output_dir, merged_file):
        """
        Merge all batch results into a single file.

        Args:
            output_dir (str): Directory containing batch files.
            merged_file (str): Path to save the merged embeddings.

        Returns:
            None
        """
        merged_embeddings = []

        # Read all batch files
        batch_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".json"))
        for batch_file in batch_files:
            with open(os.path.join(output_dir, batch_file), "r") as f:
                merged_embeddings.extend(json.load(f))

        # Save merged embeddings
        with open(merged_file, "w") as f:
            json.dump(merged_embeddings, f)

        print(f"Merged embeddings saved to {merged_file}")