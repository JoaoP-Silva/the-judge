# _the judge_
__the judge__ is a sentence bert model fine tuned to choose the sentence from a input text, that better responds to a query. At inference time, the model is binded to C++ for embeddings similarities calculation. Furthermore, the Flask API enables communication using HTTP requests to perform inferences with the model.

# Build and Running Instructions
The project can be built using Docker or by setting up the local environment manually. Is highly recomended using a Docker container to avoid dependencies errors.   

## Docker
To build the Docker image, run from the root dir: `docker build -t the-judge .`. After creating the image (can be a slow step), run `docker run -d -p 5000:5000 the-judge` to run the container in the background, exposing the port 5000 to the user.

## Manual Setup
To configure your local environment, you must first install the project's dependencies:
```bash
apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    make \
    g++ \
    curl \
    && apt-get clean
```
After installing system dependencies, run `pip install -r requirements.txt` to install python dependencies and `make` to compile the C++ code. To run the Flask server, execute from the root DIR `python server/app.py`.

In both scenarios (Docker or manual setup), to make a default test query to the model just run `sh test.sh` from the root DIR. Check the [input_test.json](https://github.com/JoaoP-Silva/the-judge/blob/main/data/input_test.json) file for query details.

# Training Details
The model was trained using the Stanford Questions and Answers Dataset (SQUAD). To perform the training, the TrainingModel class was defined (in the [Model.py](https://github.com/JoaoP-Silva/the-judge/blob/main/model/Model.py) file). The model was fine tuned in a Google Colab environenment using the [train_model.ipynb](https://github.com/JoaoP-Silva/the-judge/blob/main/model/train_model.ipynb). After the fine tuning, the model achieved a accuracy of __83%__ in the test set (10% higher than the base model).

# Inference Details
As for training, an [InferenceModel](https://github.com/JoaoP-Silva/the-judge/blob/main/model/Model.py) class was defined for inference. Additionally, to improve the model performance with large essays, the inference works as follows:

- The system receives an essay and a list of queries.
- The essay is parsed in smaller __contexts__.
- For each query, is computed a rank of contexts by embedding similarity.
- For each possible context, the best answer is computed. The first context that returns a valid response(i.e. with significant similarity value) is taken as correct and the query is marked as answered.

To more details, check the [Model.py](https://github.com/JoaoP-Silva/the-judge/blob/main/model/Model.py) and the [app.py](https://github.com/JoaoP-Silva/the-judge/blob/main/server/app.py) files.

# Comments and considerations
The model had good results, showing improved performance on the SQUAD dataset compared to the base model. However, when using a generic input ([input_test.json](https://github.com/JoaoP-Silva/the-judge/blob/main/data/input_test.json)) the model did not perform that good, answering some questions incorrectly and not identifying queries without answers. I believe that the model's performance can increase significantly by adding more ambiguous input texts and using larger contexts than those provided by SQUAD. 
