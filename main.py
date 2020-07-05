from try_word2vec import VectorScorer
from word2vec.parse_data import W2vData
from config import model_path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import multiprocessing
from scipy import spatial
from matplotlib import pyplot
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import streamlit as st
from os.path import isfile, join
from os import listdir


stop_words = set(stopwords.words('english'))


st.title("Custom Word2Vec using Streamlit visualizations")
st.markdown("Custom word2Vec targeted to generate appropriate word embeddings for tokens in resumes and job descriptions")

st.header("Training custom word2Vec model")

min_count = st.slider(
    label="min_count", min_value=1, max_value=6, step=1)
st.markdown(
    "This is the minimum count of any token required to be included in the vocabulary")
window = st.slider("Contextual window size", min_value=1, max_value=6, step=1)
st.markdown("The number of tokens on both sides of any token to be considered while predicting the target token or contecxtual size tokens. ")

cpu_count = multiprocessing.cpu_count()
workers = st.slider("cpu count",
                    min_value=1, max_value=cpu_count, step=1)
st.markdown("Number of workers or cpu cores which is to be used for training")
epochs = st.slider("Epochs",
                   min_value=1, max_value=50, step=1)
st.markdown("Number of epochs to train the model")
approach = st.selectbox("Which approach to use for training?", [
                        "Continous Bag of Words", "Continous Skip Gram model"])


if approach == "Continous Bag of Words":
    approach_sg = 0
else:
    approach_sg = 1

save_model_name = st.text_input("Save Model as : ")
button = st.button("Train Model")
if button:

    training_data = W2vData()
    sentences = training_data.prepare_training_text()
    model = Word2Vec(sentences, min_count=min_count, window=window,
                     sample=6e-5, alpha=0.03, min_alpha=0.007, workers=workers, sg=approach_sg)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=epochs, report_delay=1)
    model.save(model_path + '/' + save_model_name)
    st.write("Custom Word2Vec model has been trained and saved under the filename : {}".format(
        save_model_name))


onlyfiles = [f for f in listdir(model_path) if isfile(join(model_path, f))]
model_name = st.selectbox("Model name to load", onlyfiles)

visualize_button = st.button("Visualize model")
if visualize_button:
    model = Word2Vec.load(
        model_path+model_name)
    X = model[model.wv.vocab]
    pca = PCA(n_components=10)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    st.pyplot()


first_testing_text = st.text_input("Enter first testing string : ")
most_similar_first_button = st.button(
    "View most similar words for first testing string in vocabulary")
if most_similar_first_button:
    try:
        model = Word2Vec.load(model_path + model_name)
        most_similar_tokens_first = model.most_similar(
            first_testing_text.lower())
        st.write(most_similar_tokens_first)
    except KeyError as k:
        st.write("The word is not in the vocabulary")


second_testing_text = st.text_input("Enter second testing string : ")
most_similar_second_button = st.button(
    "View most similar words for second testing string in vocabulary")

if most_similar_second_button:
    try:
        model = Word2Vec.load(model_path + model_name)
        most_similar_tokens_first = model.most_similar(
            second_testing_text.lower())
        st.write(most_similar_tokens_first)
    except KeyError as k:
        st.write("The testing string is not in the vocabulary")

similarity_button = st.button("View similarity of above two tokens")
if similarity_button:
    vs = VectorScorer(
        model_path + model_name)

    test_string1_tokenized = word_tokenize(first_testing_text.lower())
    test_string1_without_stopwords = [
        word for word in test_string1_tokenized if not word in stopwords.words()]
    test_string2_tokenized = word_tokenize(second_testing_text.lower())
    test_string2_without_stopwords = [
        word for word in test_string2_tokenized if not word in stopwords.words()]
    first_string_vector = vs.create_vector(test_string1_without_stopwords)
    second_string_vector = vs.create_vector(test_string2_without_stopwords)
    similarity = vs.calculate_similarity(
        first_string_vector, second_string_vector)
    st.write("Similarity calculated is ", similarity)
    st.write("Word Embedding calculated for first testing string",
             first_string_vector)
    st.write("Word Embedding calculated for second testing string",
             second_string_vector)
