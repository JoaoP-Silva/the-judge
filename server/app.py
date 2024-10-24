from flask import Flask, request, jsonify
import os
import sys 

# import model
model_path = os.path.join(os.getcwd(), 'model')
sys.path.append(model_path)
from Model import InferenceModel
from Utils import extract_sentences, extract_contexts

# No answer token
NO_ANSWER = '[NO_ANSWER]'

app = Flask(__name__)

# init the model
model_path = 'model/models/trained_all_answers'
no_answer_bound = 0.5
model = InferenceModel(model_path, no_answer_bound)

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
            # rank contexts by similarity from the current query
            ranked_contexts = model._rank_answers(query, contexts, entropy = False)

            # iterate over all possible contexts untill find a valid answer
            for context in ranked_contexts:
                sentences = extract_sentences(context)
                res = model._rank_answers(query, sentences, entropy = True)[0]
                
                # if the model generated a valid answer, stop iteration
                if(res != NO_ANSWER): break

            answers.append(res)

        return jsonify({'answers': answers})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)