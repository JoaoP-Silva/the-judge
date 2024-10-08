import torch
import torch.nn as nn
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction


class TrainModel(nn.Module):
    """
    Class to abstract SBERT model training operations.
    """
    def __init__ (self, num_epochs : int, train_dataset : Dataset, test_dataset : Dataset, model_name='sentence-transformers/all-mpnet-base-v2',
                  max_length=512, batch_size = 16, save_path='model/models'):
        super(TrainModel, self).__init__()

        self._max_length = max_length
        self._model = SentenceTransformer(model_name)
        self._num_epochs = num_epochs
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._save_path = save_path
        self._loss = CoSENTLoss 
        self._batch_size = batch_size
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model.to(self._device)


    def _train(self) ->  None:
        """
        Train and eval the model.
        """
        self._model.train()

        args = SentenceTransformerTrainingArguments(
            output_dir=self._save_path,
            num_train_epochs=self._num_epochs,
            per_device_train_batch_size=self._batch_size,
            per_device_eval_batch_size=self._batch_size,
        )

        #eval the base model
        evaluator = EmbeddingSimilarityEvaluator(
                    sentences1=self._test_dataset['sentences1'],
                    sentences2=self._test_dataset['sentences2'],
                    scores=self._test_dataset['score'],
                    main_similarity=SimilarityFunction.COSINE,
                )
        results = evaluator(self._model)
        self._print_evaluation_results(results, 'base-model')
        # initialize trainer and train the model
        trainer = SentenceTransformerTrainer(
            model=self._model,
            args=args,
            train_dataset = self._train_dataset,
            eval_dataset = self._test_dataset,
            loss= CoSENTLoss,
            evaluator=evaluator,
        )
        trainer.train()

        # eval trained model
        results = evaluator(self._model)
        self._print_evaluation_results(results, 'trained-model')
        
    
    def _print_evaluation_results(self, results : dict[str, float], name : str):
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

        
    def _save_model(self):
        """
        Save the SentenceBert model in the specified path.
        """
        self._model.save_pretrained(self._save_path + '/trained_model')