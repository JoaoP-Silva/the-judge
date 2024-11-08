#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "SentenceEmbedding.h"


namespace py = pybind11;

PYBIND11_MODULE(SentenceEmbedding, m) {
    py::class_<SentenceEmbedding>(m, "SentenceEmbedding")
        // class constructor
        .def(py::init<const std::string&, const std::vector<float>&>())
        // access methods
        .def("getSentence", &SentenceEmbedding::getSentence)
        .def("getEmbedding", &SentenceEmbedding::getEmbedding)

        // compute cosine similarity from vectors
        .def("computeBestAnswer", &SentenceEmbedding::computeBestAnswer,
             py::arg("sentence_embeddings_vector"))
        
        .def("rankAnswers", &SentenceEmbedding::rankAnswers,
             py::arg("sentence_embeddings_vector"));
             
}
