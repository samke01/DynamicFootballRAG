def save_embeddings_to_faiss(embeddings, chunks, ids, embedding_model, index_name="faiss_index"):
    """
    Saves normalized embeddings, chunks, and IDs into a FAISS index for cosine similarity.

    Args:
        embeddings (np.ndarray): Precomputed embeddings.
        chunks (list): List of text chunks.
        ids (list): List of document IDs.
        embedding_model: Preloaded OpenAIEmbeddings model.
        index_name (str): Name of the FAISS index to save.
    """
    # Step 1: Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Step 2: Initialize FAISS index and add normalized embeddings
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    # Step 3: Prepare LangChain metadata and docstore
    documents = [Document(page_content=chunk, metadata={"id": id_}) for chunk, id_ in zip(chunks, ids)]
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    # Step 4: Wrap FAISS index with LangChain
    vectorstore = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embedding_model
    )

    # Step 5: Save the index
    vectorstore.save_local(index_name)
    print(f"FAISS index saved as '{index_name}' with cosine similarity enabled.")


# Batch embedding generation for all chunks
def generate_batch_embeddings(text_chunks, embedding_model, batch_size=32):
    """
    Generate embeddings for a list of text chunks in batches.

    Args:
        text_chunks (list): List of text chunks to embed.
        embedding_model (OpenAIEmbeddings): OpenAI Embeddings model.
        batch_size (int): Number of chunks to process per batch.

    Returns:
        np.ndarray: Generated embeddings.
    """
    embeddings = []
    for i in tqdm(range(0, len(text_chunks), batch_size)):
        batch = text_chunks[i:i+batch_size]
        try:
            batch_embeddings = embedding_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {e}")
            embeddings.extend([[0] * 1536] * len(batch))  # Placeholder for failed batches
    return np.array(embeddings)
