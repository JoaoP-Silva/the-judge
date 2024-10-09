#include "SentenceEmbedding.h"

// constructors
SentenceEmbedding::SentenceEmbedding(const std::string& sen, const std::vector<float>& emb)
{this->_sentence = sen; this->_embedding = emb;}


// access methods
const std::string SentenceEmbedding::getSentence() const { return this->_sentence; }
const std::vector<float> SentenceEmbedding::getEmbedding() const { return this->_embedding; }


/*
computeCosSimilarity

description:    Computes the cosine similarity between the object embedding and every embedding in
                the parameter vector.

args :          std::vector<SentenceEmbedding> senEmbVector - A vector of sentenceEmbeddings
                to compute cosine similarity.

return :        A touple with the most similar object index and the sim value.
*/
std::tuple<int, float> SentenceEmbedding::computeCosSimilarity(const std::vector<SentenceEmbedding>& senEmbVector)
{
    float best_val = std::numeric_limits<float>::max();
    int   best_idx = -1;

    std::vector<float> my_embedding = this->_embedding;
    
    // index to return
    int i = 0;
    // iterate over all embeddings
    for(auto it = senEmbVector.begin(); it < senEmbVector.end(); it++, i++)
    {
        std::vector<float> candidate_embedding = it->getEmbedding();
        float val = this->cosine_similarity(my_embedding, candidate_embedding);
        
        if(val < best_val){ best_val = val; best_idx = i;} 
    }
    
    return std::make_tuple(best_idx, best_val);
}


/*
cosine_similarity

description:    Computes the cosine similarity between two float vectors.

args :          Two vectors vector<float> vector1 and vector<float> vector2.

return :        A float with the computed cosine similarity.
*/
float SentenceEmbedding::cosine_similarity(const std::vector<float>& vector1, const std::vector<float>& vector2)
{
    if (vector1.size() != vector2.size()) 
        throw std::invalid_argument("The two vectors must have the same size.");

    float dot_product = 0.0;
    float magnitude1 = 0.0;
    float magnitude2 = 0.0;

    for (std::size_t i = 0; i < vector1.size(); ++i) {
        dot_product += vector1[i] * vector2[i];
        magnitude1 += vector1[i] * vector1[i];
        magnitude2 += vector2[i] * vector2[i];
    }

    float magnitude = std::sqrt(magnitude1) * std::sqrt(magnitude2);
    return (magnitude == 0) ? 0 : dot_product / magnitude;

}