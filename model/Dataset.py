import re
import torch
import json
import numpy as np
from datasets import Dataset

from Utils import extract_sentences

# No answer token
NO_ANSWER = '[NO_ANSWER]'


class SquadDataset_training(Dataset):
    """
    Class to abstract access and operations in the SQUAD dataset.
    The data (train and test set) is prepared to feed the model in training stage.
    """
    def __init__(self, train_path : str, test_path : str, train_size = 0, test_size = 0):
        self._train_data = self._process_data(train_path, size = train_size)
        self._test_data = self._process_data(test_path, size = test_size)
        # size of each data split (0 to use full size)
        self._train_size = train_size
        self._test_size = test_size


    def _process_data(self, path : str, size : int) -> Dataset:
        """
        Process data from the JSON file in path.
        """

        #if size <= 0, use dataset full size
        if size <= 0 : 
            size = np.inf

        # question strings
        questions = [] 
        # sentence strings
        sentences = []
        # sentence labels (1 for right answers and 0 otherwise)
        labels = []
        # context strings
        contexts = []
        # concatenated question+context strings
        questions_and_contexts = []

        with open(path, "r", encoding="utf-8") as f:
            data_to_process = json.load(f)['data']

        counter = 0
        stop = False

        for data in data_to_process:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                
                # avoid broken contexts
                if(context == ''):
                        continue
                
                curr_sentences = extract_sentences(context) 
                # set the last sentence as the [NO_ANSWER] token
                curr_sentences.append(NO_ANSWER)
                
                # iterate over questions and answers of the paragraph
                for qas in paragraph['qas']:
                    if qas['is_impossible'] == True:
                        # whether the question dont have answers, set the [NO_ANSWER] token as the right answer
                        right_answers_idx = [len(curr_sentences) - 1]
                    
                    elif qas['is_impossible'] == False:
                        # whether the question have answers, iterave over the answers populating the dataset (ignore duplicates)
                        right_answers_set = {self._get_right_sentence(answer['text'], curr_sentences) for answer in qas['answers']}

                    right_answers_idx = list(right_answers_set)
                    
                    # select only a handful of wrong sentences to mantain the dataset balanced
                    wrong_answers_idx = [x for x in range(len(curr_sentences)) if x not in right_answers_set]
                    num_right_answers = len(right_answers_idx)
                    num_wrong_answers = len(wrong_answers_idx)
                    
                    # same number of wrong and right answers
                    if(num_wrong_answers >= num_right_answers):
                        wrong_answers_idx = wrong_answers_idx[:num_right_answers]

                    # get sentences to be added in the dataset
                    indices = right_answers_idx + wrong_answers_idx
                    processed_sentences = [curr_sentences[i] for i in indices]
                    
                    curr_labels = self._get_labels(processed_sentences, num_right_answers)
                    question_arr = [qas['question']] * len(processed_sentences)
                    context_arr = [context] * len(processed_sentences)
                    # save question and context as one string
                    question_and_context = [qas['question'] + context] * len(processed_sentences)

                    questions.extend(question_arr)
                    sentences.extend(processed_sentences)
                    labels.extend(curr_labels)
                    contexts.extend(context_arr)
                    questions_and_contexts.extend(question_and_context)

                    # update counter and check if len equals size
                    counter += len(processed_sentences)
                    if counter >= size:
                        stop = True
                    
                    #whether stop flag is high, strop processing
                    if(stop): break
                if(stop): break
            if(stop): break
                            
        dataset = {
                'questions' : questions,
                'contexts' : contexts,
                'questions_and_contexts' : questions_and_contexts,
                'sentences' :sentences,
                'labels' : [1.0 if label else -1.0 for label in labels] # 1 for right answers, -1 otherwise
        }
        dataset = Dataset.from_dict(dataset)
        return dataset 


    def _get_right_sentence(self, answer : str, sentences : list[str]) -> int:
        """
        Return the sentence index that matches the position (in the original context) pos. 
        """
        for i, sentence in enumerate(sentences): 
            if answer in sentence:
                return i            
        return 0


    def _get_labels(self, sentences : list[str], num_right_answers : list[int]) -> list[bool]: 
        """
        Return the labels of sentences. True for right answers and False for wrong answers. 
        """
        # initialize the return list
        labels = [False] * len(sentences)
        labels[:num_right_answers] = [True] * num_right_answers

        return labels


class SquadDataset_inference(Dataset):
    """
    Class to abstract access and operations in the SQUAD dataset.
    The data (train and test set) is prepared to feed the model in inference stage. 
    """
    def __init__(self, train_path : str, test_path : str, train_size = 0, test_size = 0):
        self._train_data = self._process_data(train_path, size = train_size)
        self._test_data = self._process_data(test_path, size = test_size)
        # size of each data split (0 to use full size)
        self._train_size = train_size
        self._test_size = test_size

    def _process_data(self, path : str, size : int) -> Dataset:
        """
        Process data from the JSON file in path.
        """
        #if size <= 0, use dataset full size
        if size <= 0 : 
            size = np.inf

        # question strings
        questions = [] 
        # right answers strings
        answers = []
        # context strings
        contexts = []

        with open(path, "r", encoding="utf-8") as f:
            data_to_process = json.load(f)['data']

        counter = 0
        stop = False

        for data in data_to_process:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                
                # avoid broken contexts
                if(context == ''):
                        continue
                
                curr_sentences = extract_sentences(context) 
                # set the last sentence as the [NO_ANSWER] token
                curr_sentences.append(NO_ANSWER)
                
                # iterate over questions and answers of the paragraph
                for qas in paragraph['qas']:
                    if qas['is_impossible'] == True:
                        # whether the question dont have answers, set the [NO_ANSWER] token as the right answer
                        right_answers_idx = [len(curr_sentences) - 1]
                    
                    elif qas['is_impossible'] == False:
                        # whether the question have answers, iterave over the answers populating the dataset (ignore duplicates)
                        right_answers_set = {self._get_right_sentence(answer['text'], curr_sentences) for answer in qas['answers']}

                    right_answers_idx = list(right_answers_set)

                    # for every question, save only the right answers.
                    right_answers = [curr_sentences[i] for i in right_answers_idx]
                    
                    questions.append(qas['question'])
                    answers.append(right_answers)
                    contexts.append(context)

                    # update counter and check if len equals size
                    counter += len(questions)
                    if counter >= size:
                        stop = True
                    
                    #whether stop flag is high, strop processing
                    if(stop): break
                if(stop): break
            if(stop): break
                            
        dataset = {
                'questions' : questions,
                'contexts' : contexts,
                'answers' : answers
        }
        dataset = Dataset.from_dict(dataset)
        return dataset 