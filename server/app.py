from flask import Flask, request, jsonify
import os
import sys 

# import model
model_path = os.path.join(os.getcwd(), 'model')
sys.path.append(model_path)
from Model import InferenceModel
from Utils import extract_sentences, extract_contexts

app = Flask(__name__)

# init the model
model_path = 'model/models/trained_all_answers'
model = InferenceModel(model_path)

@app.route('/answers', methods=['POST'])
def answer_queries():
    try:
        # get data from json in /answers
        data = request.get_json()
        essay = data['essay']
        queries = data['queries']

        # parsing the essay in contexts
        contexts = extract_contexts(essay)

        answers = []
        for query in queries:
            # get the best context
            context = model._rank_answers(query, contexts, entropy = False)[0]
            # get the answer from it
            sentences = extract_sentences(context)
            res = model._rank_answers(query, sentences, entropy = True)[0]
            answers.append(res)

        return jsonify({'answers': answers})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)