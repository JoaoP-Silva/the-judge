import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from scipy.stats import pearsonr, spearmanr

from SentenceEmbedding import SentenceEmbedding
from Utils import extract_sentences


# No answer token
NO_ANSWER = '[NO_ANSWER]'


class TrainModel(nn.Module):
    """
    Class to abstract SBERT model training operations.
    """

    def __init__ (self, num_epochs : int, train_dataset : Dataset, test_dataset : Dataset, model_name='sentence-transformers/all-mpnet-base-v2',
                  max_length=512, batch_size = 16, save_path='model/models'):
        
        super(TrainModel, self).__init__()

        self._num_epochs = num_epochs
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._model = self._load_model(model_name)
        self._max_length = max_length
        self._batch_size = batch_size
        self._save_path = save_path
        self._loss = CoSENTLoss 
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model.to(self._device)


    def _load_model(self, path : str) -> SentenceTransformer:
            """
            Load the pre-trained SentenceTransformers located in the specified path.
            """
            return (SentenceTransformer(path))
    

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
            eval_strategy="steps",
            eval_steps=100,
        )
        trainer.train()

        # eval trained model
        results = evaluator(self._model)
        self._print_evaluation_results(results, 'trained-model')
        
    
    def _print_evaluation_results(self, results : dict[str, float], name : str) -> None:
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

        
    def _save_model(self) -> None:
        """
        Save the SentenceBert model in the specified path.
        """
        self._model.save_pretrained(self._save_path + '/trained_model')

    # acess methods
    def _set_train_dataset(self, dataset : Dataset) -> None:
        """
        Set the train dataset.
        """
        self._train_dataset = dataset

    def _set_test_dataset(self, dataset : Dataset) -> None:
        """
        Set the test dataset
        """
        self._test_dataset = dataset



class InferenceModel(nn.Module):
    """
    Class to abstract inference operations.
    """

    def __init__(self, model_name : str):

        super(InferenceModel, self).__init__()

        self._model = self._load_model(model_name)


    def _load_model(self, path : str) -> SentenceTransformer:
        """
        Load the pre-trained SentenceTransformers located in the specified path.
        """
        return (SentenceTransformer(path))
    
    def _encode(self, sentence : str) -> SentenceEmbedding:
        """
        Encode a sentence using the loaded model. Returns a SentenceEmbedding object.
        """
        embedding = self._model.encode(sentence).tolist()
        return SentenceEmbedding(sentence, embedding)


    def _compute_answer(self, query : str, context : str) -> tuple[str, int] :
        """
        Compute the answer for a given query and context.
        """
        # process the phrases and generate the SentenceEmbedding list
        sentences = extract_sentences(context)
        #add the no answer token 
        sentences.append(NO_ANSWER)

        sentences_embeddings = [self._encode(s) for s in sentences]

        # generate the input sentence_embedding
        input_str = query + context
        input = self._encode(input_str)
        # compute answer
        i, sim = input.computeBestAnswer(sentences_embeddings)
        
        return (sentences[i], sim)

    def _test_inference(self, inference_dataset : Dataset):
        """
        Compute and print the model performance on an inference dataset.
        The dataset needs to be formatted so that each line has a question, 
        a context, and a list of correct answers.
        """
        # nparray with the model results. True for right answers and False otherwise.
        hits = np.full(len(inference_dataset), 0.0, dtype=float)
        # nparray to save all similatiries to further evaluation 
        similarities = np.zeros(len(inference_dataset), dtype=float)
        
        for i, example in enumerate(inference_dataset):
            question = example['questions']
            context = example['contexts']   
            answers = example['answers']
            # convert list to set to fast 'in' operation
            right_answers = set(answers)

            # run the model
            model_answer, sim = self._compute_answer(question, context)
            
            # the model choose one of the rigth answers
            if(model_answer in right_answers): hits[i] = True
            # update similarities arr
            similarities[i] = sim
            
        accuracy = np.sum(hits) / len(hits)
        pearson_corr, _ = pearsonr(hits, similarities)
        spearman_corr, _ = spearmanr(hits, similarities)

        print(f"Model accuracy: {accuracy:.2f}")
        print(f"Pearson corr: {pearson_corr:.2f}")
        print(f"Spearman corr: {spearman_corr:.2f}")