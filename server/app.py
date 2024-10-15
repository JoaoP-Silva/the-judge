from flask import Flask, request, jsonify
import os
import sys 

# import model
model_path = os.path.join(os.getcwd(), 'model')
sys.path.append(model_path)
from Model import InferenceModel
from Utils import extract_sentences


app = Flask(__name__)

@app.route('/answers', methods=['POST'])
def answer_queries():
    try:
        # get data from json in /answers
        data = request.get_json()
        context = data['essay']
        queries = data['queries']

        # parsing context in sentences
        sentences = extract_sentences(context)
        
        # init the model
        model_path = 'model/models/trained_all_answers'
        no_answer_bound = 0.3
        model = InferenceModel(model_path, no_answer_bound)

        answers = []
        for query in queries:
            res, _ = model._compute_answer(query, sentences)
            answers.append(res)

        return jsonify({'answers': answers})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)