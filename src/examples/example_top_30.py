import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from nmf import SparseNMF

"""
This file is an example of application of Non negative matrix factorization to topic modeling.
The dataset is a 9394 x 19528 word count matrix, a column of the matrix represents a document and each row corresponds to a word.
"""

def load_dataset():
    mat_data = scipy.io.loadmat('data/tdt2_top30.mat')
    X = mat_data['X'].T
    # Extract word labels (they might be under different variable names)
    # Common names: 'fea', 'words', 'vocab', 'terms'
    if 'words' in mat_data:
        words = [w[0] if isinstance(w, np.ndarray) else w for w in mat_data['words'].flatten()]
    elif 'fea' in mat_data:
        words = [w[0] if isinstance(w, np.ndarray) else w for w in mat_data['fea'].flatten()]
    else:
        print("Available variables in .mat file:", [k for k in mat_data.keys() if not k.startswith('__')])
        words = [f'Word_{i}' for i in range(X.shape[0])] 
    return X, words


def fit_model(rank: int, solver: str , beta: float = 2):
    V, words = load_dataset()
    model = SparseNMF(V, rank)
    model.fit(solver, beta)
    return model, words


def plot_topic_words(topic_idx, W, words, top_n=5):
    """Plot top N words for a given topic"""
    top_indices = np.argsort(-W[:, topic_idx])[:top_n]
    top_words = [words[i] for i in top_indices]
    top_weights = W[top_indices, topic_idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_weights)
    plt.yticks(range(top_n), top_words)
    plt.xlabel('Weight')
    plt.title(f'Topic {topic_idx + 1}: Top {top_n} Words')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Function to plot multiple topics in a grid
def plot_all_topics(W, words, topics_to_show, top_n):
    """Plot multiple topics in a grid"""
    n_topics = len(topics_to_show)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, topic_idx in enumerate(topics_to_show):
        if idx >= len(axes):
            break
        
        top_indices = np.argsort(-W[:, topic_idx])[:top_n]
        top_words = [words[i] for i in top_indices]
        top_weights = W[top_indices, topic_idx]
        
        axes[idx].barh(range(top_n), top_weights)
        axes[idx].set_yticks(range(top_n))
        axes[idx].set_yticklabels(top_words, fontsize=8)
        axes[idx].set_xlabel('Weight', fontsize=8)
        axes[idx].set_title(f'Topic {topic_idx + 1}', fontsize=10)
        axes[idx].invert_yaxis()
    
    # Hide unused subplots
    for idx in range(n_topics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('topics_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_example():
    number_of_topics = 30
    model, words = fit_model(number_of_topics, solver="ALS", beta=2)
    # Plot a single topic (e.g., Topic 1: Clinton-Lewinsky scandal)
    plot_topic_words(19, model.W, words, top_n=10)

    plot_all_topics(model.W, words, topics_to_show=[i for i in range(10)], top_n=10)

    # Print topics as text
    print('\nTop 10 words per topic:')
    for i in range(number_of_topics):
        top_indices = np.argsort(-model.W[:, i])[:10]
        top_words = [words[idx] for idx in top_indices]
        print(f'\nTopic {i+1}: {", ".join(top_words)}')
    
if __name__ == "__main__":
    run_example()
        
