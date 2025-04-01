from difflib import SequenceMatcher, get_close_matches
import nltk
import spacy
import re
import inflect
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import load_dataset
import json
from sentence_transformers import SentenceTransformer, util
import errant
import spacy as errant_spacy
from transformers import BertTokenizer, BertModel
import torch

from correctnessPipeline import correctness_pipeline

gec_tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
gec_model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")


from wordfreq import word_frequency
from transformers import pipeline

DEBUG = True 

nltk.download('words')

from nltk.corpus import words

word_list = set(words.words())
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

def tokenize(text):
    return re.findall(r"\b\w+\b|[^\w\s]", text)

def is_misspelled(word):
    return word.lower() not in word_list

def suggest(word):
    suggestions = get_close_matches(word, word_list, n=5, cutoff=0.8)
    if suggestions:
        return max(suggestions, key=lambda w: word_frequency(w, 'en'))
    return word

def reduce_repeated_letters(word):
    candidate = word
    for _ in range(3):
        new_candidate = re.sub(r'(.)\1+', lambda m: m.group(1) * (len(m.group(0)) - 1), candidate)
        if new_candidate == candidate:
            break
        candidate = new_candidate
        if candidate.lower() in word_list:
            return candidate
    return candidate

def apply_grammar_rules(token, index):
    return None 

def apply_grammar_fixes(sentence):
    return sentence

def context_suggest(sentence, index, original_word):
    tokens = tokenize(sentence)
    if index < 0 or index >= len(tokens):
        return original_word, 0
    tokens[index] = "[MASK]"
    masked_sentence = " ".join(tokens)
    try:
        suggestions = fill_mask(masked_sentence)
        for suggestion in suggestions:
            predicted = suggestion['token_str'].strip()
            score = suggestion['score']
            if predicted.lower() in word_list and predicted.lower() != original_word.lower():
                return predicted, score
    except:
        pass
    return original_word, 0

def correct_sentence(sentence):
    corrected = correctness_pipeline(sentence)

    for i in range(len(corrected) - 1):
        if corrected[i] == "a":
            j = i + 1
            while j < len(corrected) and not re.match(r"\w+", corrected[j]):
                j += 1
            if j < len(corrected) and corrected[j][0] in "aeiou":
                if DEBUG:
                    print(f"[Rule] Article correction: 'a' before vowel -> 'an'")
                corrected[i] = "an"

    result = "".join([
        (" " + token if re.match(r"\w+", token) and i != 0 else token)
        for i, token in enumerate(corrected)
    ])
    return result

def iterative_correct(sentence, max_iter=5):
    """
    Iteratively apply correct_sentence until no further changes occur or max iterations are reached.
    """
    current_sentence = sentence
    for _ in range(max_iter):
        corrected = correct_sentence(current_sentence)
        if corrected == current_sentence:
            break
        current_sentence = corrected
    return current_sentence


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()
