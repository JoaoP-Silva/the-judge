import json

if __name__ == '__main__':
    fpath = 'essay.txt'
    with open(fpath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    queries = [
        "What is courage?",
        "What is bravery?",
        "An example of a character in the literature who displays courage",
        "An example of a character in the literature who exhibits bravery",
        "What risks a courageous act entails?",
        "What risks a brave act entails?",
	    "Who is Joao Pedro Fernandes?"
    ]

    data = {
        'essay' : text,
        'queries' : queries
    }
    json_data = json.dumps(data, indent = 4)

    with open('input_test.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)
