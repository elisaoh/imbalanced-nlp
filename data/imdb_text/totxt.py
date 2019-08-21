import json
data_json = '/Users/elisa/Downloads/nlp/sentimentAnalysis/sentiment_analysis/data/imdb/train.json'
data_txt = '/Users/elisa/Downloads/nlp/sentimentAnalysis/sentiment_analysis/code/data/imdb_text/train.txt'
with open(data_txt, 'w') as f:
    json.dump(data_json, f)