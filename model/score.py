import numpy as np
import nltk
import distance

from utils import load_formulas


def score_files(path_ref, path_hyp):
    """Loads result from file and score it
    Args:
        path_ref: (string) formulas of reference
        path_hyp: (string) formulas of prediction.
    Returns:
        scores: (dict)
    """
    # load formulas
    formulas_ref = load_formulas(path_ref)
    formulas_hyp = load_formulas(path_hyp)

    assert len(formulas_ref) == len(formulas_hyp)

    # tokenize
    refs = [ref.split(' ') for _, ref in formulas_ref.items()]
    hyps = [hyp.split(' ') for _, hyp in formulas_hyp.items()]

    # score
    return {
        "BLEU-4": bleu_score(refs, hyps)*100,
        "EM": exact_match_score(refs, hyps)*100,
        "Edit": edit_distance(refs, hyps)*100
    }


def exact_match_score(references, hypotheses):
    """Computes exact match scores.
    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)
    Returns:
        exact_match: (float) 1 is perfect
    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """Computes bleu score.
    Args:
        references: list of list (one hypothesis)
        hypotheses: list of list (one hypothesis)
    Returns:
        BLEU-4 score: (float)
    """
    references = [[ref] for ref in references]  # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(
        references, hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25)
    )
    return BLEU_4


def edit_distance(references, hypotheses):
    """Computes Levenshtein distance between two sequences.
    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)
    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return d_leven / len_tot