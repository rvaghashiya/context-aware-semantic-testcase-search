"""
Benchmarking utilities for model comparison.
"""
import time
import numpy as np
from typing import Dict, List, Any
from src.evaluation.metrics import EvaluationSuite

class ModelBenchmark:
    """Benchmark different models and configurations."""

    def __init__(self):
        self.evaluator = EvaluationSuite()
        self.results = {}

    def benchmark_embedder(self, embedder, texts: List[str], model_name: str) -> Dict[str, Any]:
        """Benchmark an embedder model."""
        print(f"Benchmarking {model_name} embedder...")

        start_time = time.time()

        # Initialize model
        embedder.initialize()
        init_time = time.time() - start_time

        # Generate embeddings
        embed_start = time.time()
        embeddings = embedder.embed_texts(texts)
        embed_time = time.time() - embed_start

        total_time = time.time() - start_time

        results = {
            'model_name': model_name,
            'n_texts': len(texts),
            'embedding_dim': embeddings.shape[1],
            'init_time': init_time,
            'embed_time': embed_time,
            'total_time': total_time,
            'texts_per_second': len(texts) / embed_time if embed_time > 0 else 0,
            'embedding_shape': embeddings.shape
        }

        print(f"{model_name}: {len(texts)} texts in {total_time:.2f}s")
        return results

    def benchmark_clustering(self, embeddings: np.ndarray, method: str, 
                           clustering_engine, **kwargs) -> Dict[str, Any]:
        """Benchmark a clustering method."""
        print(f"Benchmarking {method} clustering...")

        start_time = time.time()
        labels = clustering_engine.cluster_embeddings(embeddings, method=method, **kwargs)
        cluster_time = time.time() - start_time

        # Evaluate clustering
        eval_results = self.evaluator.evaluate_clustering(embeddings, labels)

        results = {
            'method': method,
            'cluster_time': cluster_time,
            'n_clusters': eval_results['n_clusters'],
            'metrics': eval_results['metrics'],
            'cluster_stats': eval_results['cluster_size_stats']
        }

        return results

    def run_full_benchmark(self, embedders: Dict[str, Any], texts: List[str]) -> Dict[str, Any]:
        """Run full benchmark comparing multiple models."""
        results = {'embedders': {}, 'clustering': {}}

        # Benchmark embedders
        for name, embedder in embedders.items():
            results['embedders'][name] = self.benchmark_embedder(embedder, texts, name)

        print("\nBenchmark Results Summary:")
        print("-" * 50)

        for name, result in results['embedders'].items():
            print(f"{name}: {result['texts_per_second']:.1f} texts/sec, "
                  f"dim={result['embedding_dim']}")

        return results
