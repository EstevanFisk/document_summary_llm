import logging
# We use the direct path to avoid the folder-naming confusion
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self):
        """
        Initialize the retriever builder with Google Gemini embeddings.
        This replaces the IBM WatsonX embeddings.
        """
        logger.info("Initializing Google Gemini Embeddings...")
        
        if not settings.GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY is missing in settings!")
            raise ValueError("GOOGLE_API_KEY is required for RetrieverBuilder")

        # Using the latest Gemini embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.GOOGLE_API_KEY
        )
        logger.info("Gemini Embeddings initialized successfully.")
        
    def build_hybrid_retriever(self, docs):
        """
        Build a hybrid retriever using BM25 (keyword) and Chroma (semantic) retrieval.
        """
        try:
            if not docs:
                logger.warning("No documents provided to build_hybrid_retriever.")
                return None

            # 1. Create Chroma vector store
            # The 'settings.CHROMA_DB_PATH' tells it where to save the data on your disk
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=settings.CHROMA_DB_PATH
            )
            logger.info(f"Vector store created/loaded at {settings.CHROMA_DB_PATH}")
            
            # 2. Create BM25 retriever (Keyword search)
            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever initialized.")
            
            # 3. Create Vector-based retriever (Semantic search)
            # 'k' is the number of chunks to pull; we pull it from your settings
            vector_retriever = vector_store.as_retriever(
                search_kwargs={"k": settings.VECTOR_SEARCH_K}
            )
            logger.info(f"Vector retriever initialized (k={settings.VECTOR_SEARCH_K}).")
            
            # 4. Combine them into an Ensemble (Hybrid) Retriever
            # This balances keyword matching with semantic meaning
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS
            )
            
            logger.info(f"Hybrid retriever built with weights: {settings.HYBRID_RETRIEVER_WEIGHTS}")
            return hybrid_retriever

        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {str(e)}")
            raise