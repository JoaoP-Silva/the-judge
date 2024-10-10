SRC_DIR = cpp/src
OBJ_DIR = cpp/obj
INC_DIR = cpp/include

MODULE_NAME = SentenceEmbedding
BINDINGS = bindings
OUTPUT = model/$(MODULE_NAME).so

# compiler flags
CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++20 -fPIC -I$(INC_DIR) `python3 -m pybind11 --includes`

# src and obj 
SOURCES = $(SRC_DIR)/$(MODULE_NAME).cpp $(SRC_DIR)/$(BINDINGS).cpp
OBJECTS = $(OBJ_DIR)/$(MODULE_NAME).o $(OBJ_DIR)/$(BINDINGS).o

# main rule
all: $(OUTPUT)

$(OUTPUT): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(OUTPUT) `python3.12-config --ldflags`

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)/*.o $(OUTPUT)