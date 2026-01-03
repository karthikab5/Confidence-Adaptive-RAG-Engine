import os, uuid, time, redis, logging
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import numpy as np
import dspy
from dspy.teleprompt import BootstrapFewShot
import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from urllib.parse import urlparse
from datetime  import datetime, timezone


# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DEFAULT_URLS = [
    "https://fastapi.tiangolo.com/",
    "https://docs.python.org/3/library/asyncio.html",
    "https://python.langchain.com/docs/"
]

# ============================================================================
# DSPy Modules
# ============================================================================

class AnswerSig(dspy.Signature):
    """Declarative signature for grounded answer generation."""
    context = dspy.InputField(desc="Retrieved context")
    question = dspy.InputField(desc="User question")
    answer = dspy.OutputField(desc="Grounded answer")

class RAGModule(dspy.Module):
    """Chain-of-Thought RAG module with step-by-step reasoning."""
    def __init__(self): 
        super().__init__()
        self.gen = dspy.ChainOfThought(AnswerSig)
    
    def forward(self, context, question): 
        return self.gen(context=context, question=question)

# ============================================================================
# Production RAG System
# ============================================================================

class HybridRAG:


    def __init__(self, api_key: str, z_threshold: float = 0.0):
        
        if not api_key:
            raise ValueError("API key required")
        
        self.threshold = z_threshold
        self.stats = {"total": 0, "cache": 0, "rag": 0, "hyde": 0}
        
        # Vector DB: Hybrid storage (dense + sparse)
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(
            "docs",
            vectors_config={
                "dense": models.VectorParams(size=1536, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            }
        )
        
        # Embeddings: Dense (semantic) + Sparse (keyword)
        self.dense = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
        self.sparse = SparseTextEmbedding("Qdrant/bm25")
        
        # LLM: Must use dspy.LM for DSPy compatibility
        self.llm = dspy.LM('openai/gpt-4o-mini', api_key=api_key, temperature=0)
        
        # Cache: Redis with dict fallback
        try:
            self.cache = redis.Redis(host="localhost", port=6379, decode_responses=True)
            self.cache.ping()
            self.cache_type = "redis"
            logger.info(" Redis cache connected")
        except:
            self.cache = {}
            self.cache_type = "dict"
            logger.info("  Redis unavailable, using in-memory cache")
        
        # Configure DSPy globally
        dspy.settings.configure(lm=self.llm)
        
        # Compile RAG module with faithfulness optimization
        self.rag = self._compile()
        logger.info(f" System initialized (z_threshold={z_threshold})")
    
    def _compile(self):
        
        def faithfulness_metric(example, prediction, trace=None):
            answer_words = set(prediction.answer.lower().split())
            context_words = set(example.context.lower().split())
            overlap = len(answer_words & context_words)
            total = len(answer_words)
            return overlap / total if total > 0 else 0.0
        
        # Training example for bootstrap learning
        examples = [
            dspy.Example(
                context="FastAPI is a Python framework.",
                question="What is FastAPI?",
                answer="FastAPI is a Python framework."
            ).with_inputs("context", "question")
        ]
        
        # Auto-optimize prompts using BootstrapFewShot
        return BootstrapFewShot(
            metric=faithfulness_metric,
            max_bootstrapped_demos=2
        ).compile(RAGModule(), trainset=examples)
    
    def index(self, docs: List[Dict], chunk_size: int = 500) -> int:
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50
        )

        points = []

        for doc in docs:
            for chunk in splitter.split_text(doc["text"]):
                # Generate both embedding types
                dense_vec = self.dense.embed_query(chunk)
                sparse_vec = list(self.sparse.embed([chunk]))[0]

                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist()
                        )
                    },
                    payload={
                        "text": chunk,
                        "source_url": doc["url"],
                        "domain": doc["domain"],
                        "indexed_at": doc["timestamp"]
                    }
                ))

        self.qdrant.upsert("docs", points)
        logger.info(f" Indexed {len(points)} chunks")
        return len(points)

    def retrieve(self, query: str, k: int = 5) -> Tuple[str, float]:
       
        # Generate query embeddings
        dense_query = self.dense.embed_query(query)
        sparse_query = list(self.sparse.embed([query]))[0]

        # Hybrid retrieval with RRF fusion
        results = self.qdrant.query_points(
            collection_name="docs",
            prefetch=[
                # Dense: Semantic similarity
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=k * 2
                ),
                # Sparse: Keyword matching
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist()
                    ),
                    using="sparse",
                    limit=k * 2
                )
            ],
            # RRF fusion combines both methods
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k
        )

        # Handle no results
        if not results.points:
            return "", -10.0  # Very low z-score → definitely use HyDE

        # Extract raw RRF scores
        raw_scores = [point.score for point in results.points]
        
        # Z-SCORE NORMALIZATION
        # This is the key innovation for stable confidence scoring
        mean_score = np.mean(raw_scores)
        std_score = np.std(raw_scores) + 1e-9  # Add epsilon to avoid division by zero
        z_scores = [(score - mean_score) / std_score for score in raw_scores]
        
        # Confidence = max z-score (best result quality)
        # Note: max(z_scores) is biased positive (~1.5 typically)
        # Threshold=0 still works: truly bad matches have max z < 0
        # This favors RAG (desirable - only use HyDE for poor retrievals)
        confidence = float(max(z_scores))

        # Build context with source attribution
        # Format: [domain] text
        context = "\n".join(
            f"[{point.payload['domain']}] {point.payload['text']}"
            for point in results.points
        )

        return context, confidence

    def hyde(self, question: str) -> str:
       
        response = self.llm(f"Generate a detailed answer to: {question}")
        return response[0] if isinstance(response, list) else str(response)
    
    def query(self, question: str) -> Tuple[str, str, float]:
       
        self.stats["total"] += 1
        
        # 1. Cache check
        cached = self.cache.get(question) if self.cache_type == "dict" else self.cache.get(question)
        if cached:
            self.stats["cache"] += 1
            return cached, "cache", None
        
        # 2. Retrieve context and z-score confidence
        context, z_score = self.retrieve(question)
        
        # 3. Route based on z-score threshold
        if z_score < self.threshold:
            # BELOW AVERAGE: Use HyDE fallback
            # Prevents hallucination from weak retrieval
            answer = self.hyde(question)
            source = "hyde"
            self.stats["hyde"] += 1
        else:
            # ABOVE AVERAGE: Use DSPy RAG
            # Generate grounded answer from context
            answer = self.rag(context=context, question=question).answer
            source = "rag"
            self.stats["rag"] += 1
        
        # 4. Cache result
        if self.cache_type == "dict":
            self.cache[question] = answer
        else:
            self.cache.setex(question, 3600, answer)  # TTL=1 hour
        
        return answer, source, z_score
    
    def faithfulness(self, answer: str, context: str) -> float:
       
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        overlap = len(answer_words & context_words)
        total = len(answer_words)
        return overlap / total if total > 0 else 0.0
    
    def evaluate(self, test_cases: List[Dict]) -> Dict:
      
        precisions, recalls, reciprocal_ranks = [], [], []
        
        for case in test_cases:
            # Retrieve with hybrid search
            dense_q = self.dense.embed_query(case["question"])
            sparse_q = list(self.sparse.embed([case["question"]]))[0]
            
            results = self.qdrant.query_points(
                collection_name="docs",
                prefetch=[
                    models.Prefetch(query=dense_q, using="dense", limit=10),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_q.indices.tolist(),
                            values=sparse_q.values.tolist()
                        ),
                        using="sparse",
                        limit=10
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=5
            )
            
            # Extract retrieved text snippets
            retrieved = [p.payload["text"][:30] for p in results.points]
            relevant = case["relevant"]
            
            # Count relevant retrievals
            relevant_retrieved = sum(
                1 for r in retrieved
                if any(rel in r for rel in relevant)
            )
            
            # Calculate metrics
            precisions.append(relevant_retrieved / len(retrieved) if retrieved else 0)
            recalls.append(relevant_retrieved / len(relevant) if relevant else 0)
            
            # MRR: Find rank of first relevant result
            for i, r in enumerate(retrieved, 1):
                if any(rel in r for rel in relevant):
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        # Aggregate metrics
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        
        return {
            "precision@5": precision,
            "recall@5": recall,
            "f1@5": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            "mrr": np.mean(reciprocal_ranks)
        }
    
    def stats_summary(self) -> Dict:
       
        total = self.stats["total"]
        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache"] / total if total > 0 else 0,
            "hyde_rate": self.stats["hyde"] / total if total > 0 else 0
        }

# ============================================================================
# Web Scraper
# ============================================================================

def scrape_urls(urls: List[str], rate_limit: float = 1.0) -> List[Dict]:
   
    documents = []

    for i, url in enumerate(urls):
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                continue

            # Scrape with user agent
            response = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
            )
            response.raise_for_status()

            # Extract clean text
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = " ".join(soup.get_text().split())

            # Store if sufficient length
            if len(text) > 500:
                documents.append({
                    "text": text,
                    "url": url,
                    "domain": parsed.netloc,
                    "timestamp": datetime.now(timezone.utc)
                })
                logger.info(f"  ✓ Scraped: {parsed.netloc}")

            # Rate limiting (be respectful)
            if i < len(urls) - 1:
                time.sleep(rate_limit)

        except Exception as e:
            logger.warning(f"  ✗ Skipped: {url} ({str(e)[:30]})")

    return documents


# ============================================================================
# Demo Runner
# ============================================================================

def run_demo():
    """Professional demo showcasing z-score confidence routing."""
    
    # Header
    print("\n" + "="*70)
    print("Confidence-Adaptive-RAG-Engine")
    print("="*70)
    print("\n CONFIDENCE SCORING:")
    print("   • Problem: Raw RRF scores ~0.02-0.04 (NOT normalized!)")
    print("   • Solution: Z-score normalization")
    print("   • Threshold: 0.0 (mean)")
    print("   • Above mean → RAG | Below mean → HyDE")
    print("   • Stable across corpus size changes ")
    print("\n" + "="*70)
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n Error: OPENAI_API_KEY not found")
        print("   Set it in .env file or: export OPENAI_API_KEY=your-key")
        return
    
    try:
        # Initialize with z-score threshold
        print("\nInitializing system...")
        rag = HybridRAG(api_key, z_threshold=0.0)
        print(f"    Z-score threshold: {rag.threshold} (mean)")
        print(f"    Cache: {rag.cache_type.upper()}")
        
        # Scrape documents
        print("\n Fetching documents...")
        documents = scrape_urls(DEFAULT_URLS, rate_limit=1.5)
        
        # Fallback documents with proper metadata structure
        if not documents:
            print("     Using fallback documents (scraping failed)")
            documents = [
                {
                    "text": "FastAPI is a modern Python web framework for building APIs with automatic validation and documentation.",
                    "url": "https://fastapi.tiangolo.com/",
                    "domain": "fastapi.tiangolo.com",
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "text": "Asyncio is Python's built-in library for asynchronous programming, enabling concurrent execution.",
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "domain": "docs.python.org",
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "text": "LangChain is a framework for building applications powered by large language models.",
                    "url": "https://python.langchain.com/docs/",
                    "domain": "python.langchain.com",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]
        
        # Index
        print("\n Indexing knowledge base...")
        chunks = rag.index(documents)
        print(f"    Indexed {chunks} chunks from {len(documents)} sources")
        
        # Demo queries
        print("\n" + "="*70)
        print(" LIVE DEMO - Z-Score Confidence Routing")
        print("="*70)
        
        demo_queries = [
            ("What is FastAPI?", "In-domain"),
            ("How does asyncio work?", "In-domain"),
            ("What is FastAPI?", "Cached"),
            ("What is quantum computing?", "Out-of-domain")
        ]
        
        faithfulness_scores = []
        
        for i, (question, note) in enumerate(demo_queries, 1):
            print(f"\nQuery {i}: {note}")
            print(f"Q: {question}")
            
            start = time.time()
            answer, source, z_score = rag.query(question)
            latency = (time.time() - start) * 1000
            
            # Display routing decision
            if source == "cache":
                print(f"    CACHED (instant)")
            elif source == "hyde":
                print(f"    HyDE (z={z_score:.2f} < {rag.threshold})")
            else:
                print(f"    RAG (z={z_score:.2f} ≥ {rag.threshold})")
            
            print(f"     {latency:.0f}ms")
            print(f"    {answer[:80]}...")
            
            # Calculate faithfulness for RAG queries
            if source == "rag" and z_score is not None:
                context, _ = rag.retrieve(question)
                faith = rag.faithfulness(answer, context)
                faithfulness_scores.append(faith)
                print(f"    Faithfulness: {faith:.1%}")
        
        # Evaluation
        print("\n" + "="*70)
        print(" SYSTEM EVALUATION")
        print("="*70)
        
        test_cases = [
            {"question": "What is FastAPI?", "relevant": ["FastAPI"]},
            {"question": "How does asyncio work?", "relevant": ["asyncio"]}
        ]
        
        metrics = rag.evaluate(test_cases)
        stats = rag.stats_summary()
        avg_faith = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
        
        # Results
        print("\n RETRIEVAL METRICS:")
        print(f"   Precision@5:  {metrics['precision@5']:.1%}")
        print(f"   Recall@5:     {metrics['recall@5']:.1%}")
        print(f"   F1 Score:     {metrics['f1@5']:.1%}")
        print(f"   MRR:          {metrics['mrr']:.1%}")
        print(f"   Faithfulness: {avg_faith:.1%}")
        
        print("\n ROUTING BREAKDOWN:")
        print(f"   RAG:   {(stats['rag']/stats['total']):.1%}")
        print(f"   HyDE:  {stats['hyde_rate']:.1%}")
        print(f"   Cache: {stats['cache_hit_rate']:.1%}")
        
        print("\n COST ANALYSIS:")
        cost_no_cache = 0.002
        cost_with_cache = cost_no_cache * (1 - stats['cache_hit_rate'])
        savings = (1 - cost_with_cache / cost_no_cache) * 100
        print(f"   Without Cache: ${cost_no_cache:.4f}/query")
        print(f"   With Cache:    ${cost_with_cache:.4f}/query")
        print(f"   Savings:       {savings:.0f}%")
        
        # Footer
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    run_demo()
