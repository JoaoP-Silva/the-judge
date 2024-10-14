import re
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers import SentenceTransformer
from datasets import Dataset

def extract_sentences(text : str) -> list[str]:
    """
    Split text to sentences using '.', '!' and '?' as delimiters.
    """
    sentence_delimiters = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_delimiters, text)
    return sentences


def get_right_sentence(answer : str, sentences : list[str]) -> int:
        """
        Return the sentence index that matches the given answer. 
        """
        for i, sentence in enumerate(sentences): 
            if answer in sentence:
                return i            
        return 0

def evaluate_model(model : SentenceTransformer, dataset : Dataset) -> None:

    evaluator = EmbeddingSimilarityEvaluator(
                sentences1=dataset['sentences1'],
                sentences2=dataset['sentences2'],
                scores=dataset['score'],
                main_similarity=SimilarityFunction.COSINE,
            )
    results = evaluator(model)
    print_evaluation_results(results, 'base-model')
    

def print_evaluation_results(results : dict[str, float], name : str) -> None:
    """
    Print an evaluator resuts dict.
    """
    print(f"EmbeddingSimilarityEvaluator : {name}")
    
    metrics = [
        ("Cosine-Similarity", "cosine"),
        ("Manhattan-Distance", "manhattan"),
        ("Euclidean-Distance", "euclidean"),
        ("Dot-Product-Similarity", "dot")
    ]
    
    for name, key in metrics:
        pearson_key = f"pearson_{key}"
        spearman_key = f"spearman_{key}"
        
        pearson_value = results.get(pearson_key, 0)
        spearman_value = results.get(spearman_key, 0)
        
        print(f"{name:<23}: Pearson: {pearson_value:.4f} Spearman: {spearman_value:.4f}")