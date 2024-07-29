from flask import Flask, render_template, request
from konlpy.tag import Okt

import joblib
import re

app = Flask(__name__)
app.debug = True
# flask 관련 설정

okt = Okt()

model_lr = None
tfidf = None
model_nb = None
dtm_vector = None

# 1번째 바인딩
def load_lr():
    global model_lr, tfidf_vector
    model_lr = joblib.load("model/movie_lr.pkl")
    tfidf_vector = joblib.load("model/movie_lr.dtm.pkl")

def load_nb():
    global model_lr, tfidf_vector
    model_nb = joblib.load("model/movie_lr.pkl")
    tfidf_vector = joblib.load("model/movie_lr.dtm.pkl")

def tw_tokenizer(text) :
    token_ko = okt.morphs(text)
    return token_ko

#전처리 해야 하는 값 가져오는 거 
# def lt_t(text):
#     review = re.sub(r"\d+", " ", text)
#     text_vector = tfidf_vector.transform([review])
#     return text_vector
def lt_t(text):
     stopwords = ["은", "는", "이", "가"]
     review = text.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]", "")
     morph = okt.morphs(review, stem=True) # 토큰 분리
     test = " ".join(morph for morph in morphs if not morph in stopwords)
     test_dtm = dtm_vector.transfrom([test])
     return test_dtm

@app.route("/")
def index():
    menu = {
        "home": True,
        "senti": False
    }
    return render_template("home.html", menu=menu)

# 실행파일 : 무조건 App.py라고 관례 따라야
# static, templates 폴더 이름 : 무조건 templates

@app.route("/senti", methods=["GET", "POST"])
def senti() :
        menu = {
            "home": True,
            "senti": False
        }
        if request.method == "GET" :
            return render_template("senti.html", menu=menu)
        else :
            review = request.form["request"]
            review_text = lt_t(review)
            lr_result = model_lr.predict(review_text)[0]
            review_text2 = lt_nb(review)
            nb_result = model_lr.predict(review_text2)[0]
            lr = "긍정" if lr_result else "부정"
            nb = "긍정" if lr_result else "부정"
            movie = {"review":review, "lr":lr, "nb":nb}
        return render_template("senti_result.html", menu=menu, movie=movie)
# 사용자가 리뷰를 남기면
if __name__ == "__main__" :
    load_lr()
    load_nb()
    app.run()