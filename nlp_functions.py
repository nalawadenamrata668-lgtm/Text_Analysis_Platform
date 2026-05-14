#Try AI directly in your favourite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI
# WORD CLOUD
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
# WORD CLOUD
def show_wordcloud(text):
    try:
        wordcloud= WordCloud(width=800, height=400, background_color="white").generate(" ".join(text))
        # for wordcloud we pass text not tokens
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud)
        plt.axis("off")
        return plt
    except Exception as e:
        return f"Error generating word cloud:m {e}"


# N-GRAM ANALYSIS
from nltk.util import ngrams
from collections import Counter
import plotly.graph_objects as go

def plot_top_ngrams_bar_chart(tokens, gram_n=3, top_n=15):
    try:
        ngram = list(ngrams(tokens, gram_n))
        ngram_counts = Counter(ngram).most_common(top_n)

        if not ngram_counts:
            raise ValueError("No n-grams found in the given  token list")

        labels = []
        counts = []
        for biagram, count in ngram_counts:
            labels.append(" ".join(biagram))
            counts.append(count)

        # plotly bar chart
        fig = go.Figure(data=
        [go.Bar(
            x=labels,
            y=counts,
            text=counts,
            textposition="outside")])

        # update layout
        fig.update_layout(height=550,
                          title="top 15 Biagrams",
                          xaxis_title="Labels",
                          yaxis_title="Frequency")

        st.plotly_chart(fig)
    except Exception as e:
        print(f"An Error happen: {e}")


#CREATING CHUNKS
import spacy
def split_into_chunks_spacy(text, max_length=500):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    chunks = []
    current_chunk = ""

    for sent in doc.sents:
        sentence = sent.text.strip()
        if len(current_chunk) + len(sentence) <= 500:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks



#EMOTIONAL ANALYSIS
from transformers import pipeline
import pandas as pd

model_name= "j-hartmann/emotion-english-distilroberta-base"
emotion_classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None)
from collections import defaultdict


def detect_emotions(text):
    chunks = split_into_chunks_spacy(text)
    emotion_totals = {}
    emotion_count = {}
    emotion_count = defaultdict(int)

    for chunk in chunks:
        results = emotion_classifier(chunk)[0]
        for result in results:
            label = result["label"]
            score = result["score"]
            emotion_totals[label] = emotion_totals.get(label, 0) + score
            emotion_count[label] += 1

    emotion_counts = dict(emotion_count)

    emotion_averages = {label: emotion_totals[label] / emotion_counts[label] for label in emotion_totals}
    sorted_emotions = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_emotions[:5]
    df = pd.DataFrame(top_5, columns=["Emotion", "Score"])
    return df

# SENTIMENT ANALYSIS

model_name= "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_classifier= pipeline("sentiment-analysis", model= model_name,  tokenizer= model_name , return_all_scores=True)


# # CREATING FUNCTION FOR SENTIMENT ANALYSIS
# def detect_overall_sentiment_avg(text):
#     try:
#         sentiment_labels = {
#             "LABEL_0": "Negative",
#             "LABEL_1": "Neutral",
#             "LABEL_2": "Positive"
#         }
#         chunks = split_into_chunks_spacy(text)
#         score_total = {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0}
#         chunk_count = len(chunks)
#
#         for chunk in chunks:
#             results = sentiment_classifier(chunk)[0]
#             for res in results:
#                 label = sentiment_labels[res["label"]]
#                 score_total[label] += res["score"]
#
#         avg_score = {}
#         for label in score_total:
#             avg_score[label] = score_total[label] / chunk_count
#         if avg_score:
#             overall_sentiment = max(avg_score, key=avg_score.get)
#         else:
#             overall_sentiment = "No sentiment found"        return {
#             "overall_sentiment": overall_sentiment,
#             "average_scores": avg_score
#         }
#
#     except Exception as e:
#         return {f"error {e}"}


#TONE OF SPEECH CLASSIFICATION

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = [
    "factual",
    "opinion",
    "question",
    "command",
    "emotion",
    "personal experience",
    "suggestion",
    "story",
    "prediction",
    "warning",
    "instruction",
    "definition",
    "narrative",
    "news",
    "argument"
]
def classify_custom(text):
    result = classifier(text, candidate_labels=labels)
    return {
        "text": text,
        "predicted_category": result["labels"][0],
        "score": result["scores"][0],
        "all_categories": list(zip(result["labels"], result["scores"]))
    }



# TEXT SUMMARIZATION

def summarize_large_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Step 1: Split text into manageable chunks
    chunks = split_into_chunks_spacy(text, max_length=500)  # Use pysbd or whatever you prefer

    # Step 2: Summarize each chunk individually
    chunk_summaries = []
    for chunk in chunks:
        input_length = len(chunk.split())  # rough word count
        max_summary_length = min(300, max(30, int(input_length * 0.7)))  # max 70% of input or cap at 300
        min_summary_length = min(100, max(20, int(input_length * 0.3)))  # min 30% of input or cap at 100
        summary = summarizer(chunk, max_length=max_summary_length, min_length=min_summary_length, do_sample=False)[0]['summary_text']
        chunk_summaries.append(summary)

    # Step 3: Combine all chunk summaries into one text
    combined_summary_text = " ".join(chunk_summaries)

    # Optional Step 4: Summarize the combined summary for a concise result
    input_length = len(combined_summary_text.split())  # rough word count
    max_summary_length = int(input_length * 0.9)  # max 70% of input or cap at 300
    min_summary_length = int(input_length * 0.3)  # min 30% of input or cap at 100
    final_summary = summarizer(combined_summary_text, max_length=max_summary_length, min_length=min_summary_length, do_sample=False)[0]['summary_text']

    return final_summary
