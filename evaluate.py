"""
Evaluation metrics: Exact Match and F1 score.
Based on the standard SQuAD evaluation script.
"""

import re
import string
import json
import sys


def normalize_answer(s):
    """
    Normalize answer string: lowercase, remove punctuation, articles, extra whitespace.
    """
    def remove_articles(text): # remove 'a', 'an', 'the'
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text): # remove extra whitespace
        return ' '.join(text.split())
    
    def remove_punc(text): # remove punctuation
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truths):
    """Check if normalized prediction matches any ground truth."""
    # Ground truths can be a list of acceptable answers (e.g., "ans1|ans2|...")
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if norm_pred == normalize_answer(gt):
            return 1.0
    return 0.0


def f1_score(prediction, ground_truths):
    """Compute token-level F1 between prediction and best matching ground truth."""
    best_f1 = 0.0
    pred_tokens = normalize_answer(prediction).split() # tokenize the prediction
    
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split() # tokenize the ground truth
        
        common = set(pred_tokens) & set(gt_tokens) # find common tokens
        num_common = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common) # count common tokens considering duplicates
        
        if num_common == 0:
            continue
        
        precision = num_common / len(pred_tokens) if pred_tokens else 0
        recall = num_common / len(gt_tokens) if gt_tokens else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        best_f1 = max(best_f1, f1) # keep the best F1 across all ground truths
    
    return best_f1


def evaluate(predictions_path, references_path):
    """
    Evaluate predictions against references.
    
    predictions_path: text file with one prediction per line
    references_path: JSON file with list of reference answers (each can be "ans1|ans2|...")
                     OR text file with one answer per line
    """
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = [line.strip() for line in f]
    
    # Load references
    if references_path.endswith('.json'): # support JSON format for references
        with open(references_path, 'r') as f:
            raw_refs = json.load(f)
        references = []
        # An example is [["Room 723", "Rm 723"], "1868", ["Dan Klein", "D. Klein"]]
        for ref in raw_refs:
            if isinstance(ref, list): # already a list of answers
                references.append(ref) 
            elif isinstance(ref, str): # split by '|' if it's a string of answers
                references.append(ref.split('|'))
            else:
                references.append([str(ref)]) # convert non-string to string and wrap in list
    else:
        with open(references_path, 'r') as f:
            references = [line.strip().split('|') for line in f]
    
    assert len(predictions) == len(references), \
        f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
    
    em_scores = []
    f1_scores = []
    
    for pred, refs in zip(predictions, references):
        em = exact_match_score(pred, refs)
        f1 = f1_score(pred, refs)
        em_scores.append(em)
        f1_scores.append(f1)
    
    avg_em = sum(em_scores) / len(em_scores) * 100
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100
    
    print(f"Exact Match: {avg_em:.2f}%")
    print(f"F1 Score:    {avg_f1:.2f}%")
    print(f"Total:       {len(predictions)} questions")
    
    return avg_em, avg_f1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate.py <predictions_path> <references_path>")
        sys.exit(1)
    
    evaluate(sys.argv[1], sys.argv[2])
