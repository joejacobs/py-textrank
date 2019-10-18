import nltk
import numpy as np
import textblob as tb


STOPWORDS = nltk.corpus.stopwords.words("english")


def tokenize(text, pos_tags=[]):
    """
    Tokenize a string with TextBlob.
    """
    assert isinstance(text, str)
    assert isinstance(pos_tags, list)
    assert len(text) > 0, "text must not be empty"

    blob = tb.TextBlob(text, pos_tagger=tb.en.taggers.PatternTagger())

    if pos_tags != []:
        tokens = [x for x, y in blob.pos_tags
                  if y[:2] in pos_tags and x not in STOPWORDS]
    else:
        tokens = [x for x in blob.tokens if x not in STOPWORDS]

    return tokens


def pagerank(adj, d=0.85, eps=1e-4):
    """
    Run PageRank on a weighted adjacency matrix.
    """
    assert isinstance(d, float)
    assert isinstance(eps, float)
    assert isinstance(adj, np.ndarray)
    assert adj.shape[0] == adj.shape[1]
    assert adj.shape[0] > 0

    adj_norm = np.diag(1/adj.sum(axis=1)).dot(adj)
    score = np.ones((adj.shape[0],))

    while True:
        new_score = (1 - d) + (d * adj_norm.T.dot(score))
        delta = abs(new_score - score).sum()

        if delta <= eps:
            return new_score

        score = new_score


class TextRank:
    def __init__(self, cooccurrence_window_sz=2, pos_tags=[]):
        self._cooccurrence_window_sz = cooccurrence_window_sz
        self._pos_tags = pos_tags
        self._vocab_dict = {}
        self._vocab_lst = []
        self._adj_mat = None

    def _update_vocab(self, tokens):
        """
        Update vocabulary with new tokens.
        """
        new_tokens = set(tokens) - set(self._vocab_lst)

        for t in sorted(list(new_tokens)):
            self._vocab_dict[t] = len(self._vocab_lst)
            self._vocab_lst.append(t)

        n_tokens = len(self._vocab_lst)
        new_adj_mat = np.zeros((n_tokens, n_tokens))

        if self._adj_mat is not None:
            old_vocab_sz = self._adj_mat.shape[0]
            new_adj_mat[:old_vocab_sz, :old_vocab_sz] = self._adj_mat

        self._adj_mat = new_adj_mat

    def get_scores(self, damping_factor=0.85, epsilon=1e-4):
        scores = list(pagerank(self._adj_mat, damping_factor, epsilon))
        return {token: score for token, score in zip(self._vocab_lst, scores)}

    def update_adjacency_matrix(self, text):
        assert len(text) > 0, "text must not be empty"

        tokens = tokenize(text, self._pos_tags)
        self._update_vocab(tokens)
        n_tokens = len(tokens)

        for i in range(n_tokens):
            fst = min(i + 1, n_tokens)
            lst = min(i + self._cooccurrence_window_sz, n_tokens)
            query = [tokens[i]] * (lst - fst)
            window = tokens[fst:lst]

            for t0, t1 in zip(query, window):
                idx0 = self._vocab_dict.get(t0)
                idx1 = self._vocab_dict.get(t1)
                self._adj_mat[idx0, idx1] += 1
                self._adj_mat[idx1, idx0] += 1
