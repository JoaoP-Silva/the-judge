#ifndef SENTENCE_EMBEDDING_H
#define SENTENCE_EMBEDDING_H

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <limits>
#include <cmath>

class SentenceEmbedding 
{
    std::string _sentence;
    std::vector<float> _embedding; 

    public:
        // default constructor
        SentenceEmbedding(const std::string& sen, const std::vector<float>& emb);

        // access methods
        const std::string getSentence() const ;
        const std::vector<float> getEmbedding() const ;
        
        /*
        computeBestAnswer
        
        description:    Computes the cosine similarity between the object embedding and every embedding in
                        the parameter vector.
        
        args :          std::vector<SentenceEmbedding> senEmbVector - A vector of sentenceEmbeddings
                        to compute cosine similarity

        return :        A touple with the most similar object index and the sim value.
         */
        std::tuple<int, float> computeBestAnswer(const std::vector<SentenceEmbedding>& senEmbVector);

    private:
        // Private methods
        /*
        cosine_similarity
        
        description:    Computes the cosine similarity between two float vectors.
        
        args :          The two vectors std::vector<float> vector1 and std::vector<float> vector2.

        return :        The cosine similarity between the two input vectors.
         */
        float cosine_similarity(const std::vector<float>& vector1, const std::vector<float>& vector2);
        

};


# endif