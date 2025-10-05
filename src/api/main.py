"""
FastAPI backend for semantic test case clustering.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np

# Import project modules
try:
    from src.data.dataloader import load_demo_data
    from src.models.model_factory import ModelFactory
    from src.clustering.clustering_engine import ClusteringEngine
    from src.clustering.similarity_engine import SimilarityEngine
    from src.utils.config import Config
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

app = FastAPI(
    title="Semantic Test Case Clustering API",
    description="API for semantic clustering and search of software test cases",
    version="1.0.0"
)

# Global variables for caching
cached_embeddings = {}
cached_engines = {}

class EmbedRequest(BaseModel):
    texts: List[str]
    model_name: str = "bert"

class SearchRequest(BaseModel):
    query: str
    model_name: str = "bert"
    top_k: int = 10

class ClusterRequest(BaseModel):
    model_name: str = "bert"
    method: str = "kmeans"
    n_clusters: Optional[int] = 8
    n_samples: int = 500

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Semantic Test Case Clustering API",
        "version": "1.0.0",
        "available": MODULES_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "modules_available": MODULES_AVAILABLE}

@app.post("/embed")
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for texts."""
    if not MODULES_AVAILABLE:
        raise HTTPException(status_code=500, detail="Project modules not available")

    try:
        # Create embedder
        config = {'name': request.model_name, 'embedding_dim': 512}
        embedder = ModelFactory.create_embedder(request.model_name, config)
        embedder.initialize()

        # Generate embeddings
        embeddings = embedder.embed_texts(request.texts)

        return {
            "embeddings": embeddings.tolist(),
            "shape": embeddings.shape,
            "model": request.model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster")
async def cluster_test_cases(request: ClusterRequest):
    """Cluster test cases using semantic embeddings."""
    if not MODULES_AVAILABLE:
        raise HTTPException(status_code=500, detail="Project modules not available")

    try:
        # Load demo data
        df = load_demo_data(max_samples=request.n_samples)
        texts = df['test_description'].tolist()

        # Generate embeddings
        config = {'name': request.model_name, 'embedding_dim': 512}
        embedder = ModelFactory.create_embedder(request.model_name, config)
        embedder.initialize()
        embeddings = embedder.embed_texts(texts)

        # Apply clustering
        clustering_engine = ClusteringEngine()

        if request.method == "kmeans":
            labels = clustering_engine.apply_kmeans(embeddings, n_clusters=request.n_clusters)
        elif request.method == "hierarchical":
            labels = clustering_engine.apply_hierarchical(embeddings, n_clusters=request.n_clusters)
        else:  # dbscan
            labels = clustering_engine.apply_dbscan(embeddings)

        # Get statistics
        stats = clustering_engine.get_cluster_statistics(embeddings)

        return {
            "labels": labels.tolist(),
            "texts": texts,
            "statistics": stats,
            "model": request.model_name,
            "method": request.method
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_similar(request: SearchRequest):
    """Search for semantically similar test cases."""
    if not MODULES_AVAILABLE:
        raise HTTPException(status_code=500, detail="Project modules not available")

    # Mock search results for demo
    mock_results = [
        {"index": 0, "text": "test_user_login_success", "similarity": 0.89},
        {"index": 1, "text": "verify_authentication_valid", "similarity": 0.85},
        {"index": 2, "text": "check_user_signin_functionality", "similarity": 0.82},
        {"index": 3, "text": "test_login_invalid_password", "similarity": 0.78},
        {"index": 4, "text": "validate_user_credentials", "similarity": 0.75}
    ]

    return {
        "query": request.query,
        "results": mock_results[:request.top_k],
        "model": request.model_name
    }

@app.get("/models")
async def get_supported_models():
    """Get list of supported embedding models."""
    if not MODULES_AVAILABLE:
        return {"models": []}

    try:
        models = ModelFactory.get_supported_models()
        return {"models": models}
    except:
        return {"models": ["elmo", "bert", "t5"]}

@app.get("/metrics/{model_name}")
async def get_model_metrics(model_name: str):
    """Get performance metrics for a specific model."""
    mock_metrics = {
        "elmo": {
            "silhouette_score": 0.485,
            "davies_bouldin_score": 0.742,
            "calinski_harabasz_score": 156.4,
            "precision_at_10": 0.847,
            "recall_at_10": 0.823,
            "mrr": 0.756
        },
        "bert": {
            "silhouette_score": 0.521,
            "davies_bouldin_score": 0.698,
            "calinski_harabasz_score": 178.2,
            "precision_at_10": 0.892,
            "recall_at_10": 0.856,
            "mrr": 0.798
        },
        "t5": {
            "silhouette_score": 0.467,
            "davies_bouldin_score": 0.789,
            "calinski_harabasz_score": 142.1,
            "precision_at_10": 0.824,
            "recall_at_10": 0.801,
            "mrr": 0.734
        }
    }

    if model_name not in mock_metrics:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"model": model_name, "metrics": mock_metrics[model_name]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
