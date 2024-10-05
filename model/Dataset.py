import re
import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# No answer token
NO_ANSWER = '[NO_ANSWER]'
# Dataset mode (train or dev)
TRAIN_MODE = 0
DEV_MODE = 1


class SquadDataset(Dataset):
    def __init__(self, train_path, dev_path, model_name="sentence-transformers/all-MiniLM-L6-v2", max_length=512):
        self._train_path = train_path
        self._dev_path = dev_path
        self._train_data = self.process_data(self._train_path)
        self._dev_data = self.process_data(self._dev_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._max_length = max_length
        self._mode = TRAIN_MODE

    def _process_data(self, path : str):
        """
        Process data from the JSON file in path.
        """
        # question strings
        questions = [str] 
        # sentence strings
        sentences = [str]
        # sentence labels (1 for right answers and 0 otherwise)
        labels = [int]
        # context strings
        contexts = [str]

        with open(path, "r", encoding="utf-8") as f:
            data_to_process = json.load(f)['data']

        for data in data_to_process:
                for paragraph in data['paragraphs']:
                        context = paragraph['context']
                        curr_sentences = self._extract_sentences(context) 
                        # set the last sentence as the [NO_ANSWER] token
                        curr_sentences.append(NO_ANSWER)
                        
                        # iterate over questions and answers of the paragraph
                        for qas in paragraph['qas']:
                                question_arr = [qas['question']] * len(curr_sentences)
                                context_arr = [context] * len(curr_sentences)
                                
                                # whether the question dont have answers, set the [NO_ANSWER] token as the right answer
                                if qas['is_impossible'] == True or len(qas['answers']) == 0:
                                        labels = [False] * len(curr_sentences)
                                        labels[-1] = True
                                
                                # whether the question have answers, iterave over the answers populating the dataset
                                elif qas['is_impossible'] == False:
                                        positions = [pos['answer_start'] for pos in qas['answers']]
                                        # get the right sentences index over all sentences
                                        right_answers_idx = [self._get_sentence_index(position, curr_sentences) for position in positions]
                                        # all sentence labels
                                        curr_labels = self._get_labels(curr_sentences, right_answers_idx)
                                
                                questions.extend(question_arr)
                                sentences.extend(curr_sentences)
                                labels.extend(curr_labels)
                                contexts.extend(context_arr)
                
        dataset = {
                'questions' : questions,
                'sentences' :sentences,
                'labels' :labels,
                'contexts' : contexts,
        }  
        return dataset 

    def _extract_sentences(text : str) -> list[str]:
        """
        Split text to sentences using '.', '!' and '?' as delimiters.
        """
        sentence_delimiters = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_delimiters, text)
        return sentences


    def _get_sentence_index(pos : int, sentences : list[str]) -> int:
        """
        Return the sentence index that matches the position (in the original context) pos. 
        """
        curr_pos = 0
        for i, sentence in enumerate(sentences): 
            if(pos <= curr_pos):
                return i
            curr_pos += len(sentence)

    def _get_labels(sentences : list[str], answers_index : list[int]) -> list[bool]: 
        """
        Return the labels of sentences. 1 for right answers and 0 for wrong answers. 
        """
        # initialize the return list
        labels = [False] * len(sentences)
        for i in answers_index:
            if(i == None):
                continue
            else:
                labels[i] = True

        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx : int):
        # select data by the Dataset current mode (train or dev).
        if(self._mode == TRAIN_MODE):
            questions = self._train_data['questions']
            sentences = self._train_data['sentences']
            labels = self._train_data['labels']
            contexts = self._train_data['contexts']

        elif(self._mode == DEV_MODE):
            questions = self._dev_data['questions']
            sentences = self._dev_data['sentences']
            labels = self._dev_data['labels']
            contexts = self._dev_data['contexts']

        question = questions[idx]
        sentence = sentences[idx]
        label = labels[idx]
        context = contexts[idx]

        # tokenize question + context as the model input
        inputs = self._tokenizer(
            question, context,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # tokenize the sentence
        sentence_encoding = self.tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'sentence_input_ids': sentence_encoding['input_ids'].squeeze(),
                    'sentence_attention_mask': sentence_encoding['attention_mask'].squeeze(),
                    'label': torch.tensor(1 if label else -1)  # 1 for right answers, -1 otherwise
                }
