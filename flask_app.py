# A very simple Flask Hello World app for you to get started with...

from flask import Flask
from flask import jsonify
from flask import request
from flask import Response
from flask_cors import CORS
import json
from gensim.models.doc2vec import Doc2Vec
from cleantext import clean
# import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
# import pandas as pd

def remove_stopword(text):
    # nltk.download('stopwords')
    stops = set(stopwords.words("english"))
    words = [w for w in text.lower().split() if not w in stops]
    final_text = " ".join(words)

    return final_text

def remove_special_char(final_text):
    # Special Characters
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", final_text )
    review_text = re.sub(r"\'s", " 's ", final_text )
    review_text = re.sub(r"\'ve", " 've ", final_text )
    review_text = re.sub(r"n\'t", " 't ", final_text )
    review_text = re.sub(r"\'re", " 're ", final_text )
    review_text = re.sub(r"\'d", " 'd ", final_text )
    review_text = re.sub(r"\'ll", " 'll ", final_text )
    review_text = re.sub(r",", " ", final_text )
    review_text = re.sub(r"\.", " ", final_text )
    review_text = re.sub(r"!", " ", final_text )
    review_text = re.sub(r"\(", " ( ", final_text )
    review_text = re.sub(r"\)", " ) ", final_text )
    review_text = re.sub(r"\?", " ", final_text )
    review_text = re.sub(r"\s{2,}", " ", final_text )

    return review_text

def clean_data(text):
    return clean(text,
    fix_unicode=True,
    lower=True,
    no_line_breaks=False,
    no_urls=False,
    no_emails=False,
    no_phone_numbers=False,
    no_numbers=False,
    no_digits=False,
    no_currency_symbols=False,
    no_punct=True,
    lang="en"
)

def get_doc2vec_sim(disc_comments):
    explored = []
    unexplored = []
    X = []
    Y = []

    # explored = disc_comments["explored"]
    # unexplored = disc_comments["unexplored"]

    for i in disc_comments["explored"]:
        clean_text = clean_data(i["text"])
        stop_removed = remove_stopword (clean_text)
        special_removed = remove_special_char(stop_removed)
        query = special_removed.split(" ")
        explored.append({"query":query, "orig": i["text"], "id": i["id"], "disc": i["disc"]})

    for i in disc_comments["unexplored"]:
        clean_text = clean_data(i["text"])
        stop_removed = remove_stopword (clean_text)
        special_removed = remove_special_char(stop_removed)
        candidate = special_removed.split(" ")
        unexplored.append({"candidate":candidate, "orig": i["text"], "id": i["id"], "disc": i["disc"]})

    d2vmodel = Doc2Vec.load("/home/mjasim/mysite/d2v_40k_trim.model")

    # write new methods for average similarity and coverage separately and efficiently

    for i in unexplored:
        score = 0
        for j in explored:
            # print(i["candidate"], j["query"])
            temp_score = d2vmodel.n_similarity(i["candidate"], j["query"])
            if(temp_score > 0.8):
                X.append({"text": i["orig"], "score": str(score), "id": i["id"], "disc": i["disc"]})
                continue;
            score = score + temp_score

        avg_score = (score / len(explored))
        # if(score > 0.80):
        #     X.append({"text": i["orig"], "score": str(score), "id": i["id"], "disc": i["disc"]})
        # else:
        #     Y.append({"text": i["orig"], "score": str(score), "id": i["id"], "disc": i["disc"]})
        Y.append({"text": i["orig"], "score": str(avg_score), "id": i["id"], "disc": i["disc"]})

    covered = sorted(X, key = lambda i: i["score"], reverse=True)
    uncovered = sorted(Y, key = lambda i: i["score"], reverse=False)

    k = [x['text'] for x in covered]
    unique_covered = []

    for i in Counter(k):
        all = [x for x in covered if x["text"] == i]
        unique_covered.append(max(all, key=lambda x: x["score"]))

    k = [x['text'] for x in uncovered]
    unique_uncovered = []

    for i in Counter(k):
        all = [x for x in uncovered if x["text"] == i]
        unique_uncovered.append(max(all, key=lambda x: x["score"]))

    return [unique_covered, unique_uncovered]

def get_sim(disc_comments):

    # df = pd.read_json("/home/mjasim/mysite/mat_data_cc.json")

    # prop_data = df.loc[df["discussion_name"] == "Roundabout"]
    # comment_list = prop_data.comments.tolist()

    # documents = [c["comment"] for c in comment_list[0]]
    # doc = documents[17]

    documents = disc_comments["comments"]
    doc = disc_comments["query"]

    from collections import defaultdict
    from gensim import corpora

    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    from gensim import models
    lsi = models.LsiModel(corpus, id2word=dictionary)

    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space

    from gensim import similarities
    index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it

    sims = index[vec_lsi]  # perform a similarity query against the corpus

    sim_obj_list = []

    sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for i, s in enumerate(sorted_sims):
        sim_obj_list.append({"id": s[0], "comment": documents[s[0]], "score": str(s[1])})

    return sim_obj_list

def sim_try(disc_comments):
    documents = disc_comments["comments"]
    # doc = disc_comments["query"]
    return documents

app = Flask(__name__)
CORS(app)

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == 'POST':
        disc_comments = request.get_json()
        return Response(json.dumps(get_doc2vec_sim(disc_comments)), mimetype='application/json')
    else:
        return jsonify('Hello')