from time import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nmf import NonNegMatrix, SepNMF


"""
This code is a modified version of the example in the scikit-learn NMF documentation. 
The purpose of this file is to show the simplest example of application of NMF in topic modeling.
Instead of using NMF from scikit-learn my implementation is used. Particularly the separability assumption is made and the SNPA solver is used.
-------------
Reference:
https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
--------------
Notes
-----
- TF-IDF (Term Frequency — Inverse Document Frequency):
    * Term Frequency (TF) measures how often a term appears in a document.
    * Inverse Document Frequency (IDF) downweights terms that appear across many documents,
      giving higher weight to terms that are more discriminative for specific documents.
    * TF-IDF = TF * IDF. TF-IDF vectors represent documents where terms important to each
      document (relative to the corpus) receive larger values.
    * When used with NMF, TF-IDF highlights words that characterize topics and reduces
      the influence of very common words.

- Interpreting the results:
    * Each topic is represented by a row of the H matrix (coefficients for basis vectors).
    * The top-weighted words in each topic (highest values in the topic vector) are
      interpreted as the most representative terms for that topic.
    * W (the basis) contains the selected anchor columns (documents/features depending on formulation).
    * H contains coefficients to reconstruct documents as non-negative combinations of topics.

"""

def plot_top_words(model, feature_names, n_top_words, title):
    """
    Plot top words for each topic in the model.

    Parameters
    ----------
    model : SepNMF
        Fitted separable NMF model exposing H (topic vectors) and/or W.
    feature_names : array-like of str
        Mapping from feature indices to words (from vectorizer).
    n_top_words : int
        Number of top words to display per topic.
    title : str
        Plot title.
    """
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.H):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 15})
        ax.tick_params(axis="both", which="major", labelsize=10)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=20)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()



def load_dataset(n_samples: int, n_features: int):

    """
    Load the 20 newsgroups dataset and compute TF-IDF features.

    Parameters
    ----------
    n_samples : int
        Number of documents to load (from the start of the corpus).
    n_features : int
        Maximum number of features (vocabulary size) to keep; low-frequency and
        extremely common terms are filtered by min_df/max_df.

    Returns
    -------
    tfidf : sparse matrix, shape (n_samples, n_features)
        TF-IDF feature matrix for documents.
    tfidf_vectorizer : TfidfVectorizer
        Fitted vectorizer (used to obtain feature names).
    """


    # Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
    # to filter out useless terms early on: the posts are stripped of headers,
    # footers and quoted replies, and common English words, words occurring in
    # only one document or in at least 95% of the documents are removed.


    print("Loading dataset...")
    t0 = time()
    data, _ = fetch_20newsgroups(
        shuffle=True,
        random_state=1,
        remove=("headers", "footers", "quotes"),
        return_X_y=True,
    )
    data_samples = data[:n_samples]
    print("done in %0.3fs." % (time() - t0))

    # Use tf-idf features for NMF.
    # max_df removes very common words, min_df removes rare words, max_features limits the vocabulary size
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
    )
    t0 = time()
    # Fit the TF-IDF vectorizer to the data samples and transform the data into TF-IDF features
    # reference: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    m, n = tfidf.get_shape()
    sparsity = 1 - tfidf.getnnz()/float(m * n)
    print(f"sparsity of the data matrix: {sparsity}")
    return tfidf, tfidf_vectorizer

def fit_model(n_samples: int, n_features: int, n_components: int, n_top_words: int):
    """
    Run the end-to-end example: load data, fit SepNMF, and plot topics.

    Steps
    -----
    1. Load TF–IDF features with load_dataset.
    2. Convert TF–IDF sparse matrix to a dense non-negative matrix for SepNMF.
       (bear in mind memory usage; for large corpora use chunking or a sparse-aware algorithm)
    3. Instantiate SepNMF and fit using SNPA solver.
    4. Plot top words per discovered topic.

    Parameters
    ----------
    n_samples, n_features, n_components, n_top_words : ints
        Controls dataset size, vocabulary size, number of topics and words shown.
    """
    
    tfidf, tfidf_vectorizer = load_dataset(n_samples, n_features)
    V = NonNegMatrix(tfidf.toarray()) # type: ignore
    model = SepNMF(V, n_components)

    model.fit(solver="SNPA")

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    plot_top_words(model, tfidf_feature_names, n_top_words, "NMF topics with SNPA solver and Frobenius norm")


def run_example():
    n_samples = 2000
    n_features = 1000
    n_components = 10
    n_top_words = 15
    fit_model(n_samples, n_features, n_components, n_top_words)

if __name__ == "__main__":
    run_example()