import re
import nltk
from nltk.corpus import words
from difflib import get_close_matches, SequenceMatcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

gec_tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
gec_model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

from wordfreq import word_frequency
from transformers import pipeline

DEBUG = True  # Set to False to disable debug prints

# Download word list if not already downloaded
nltk.download('words')

# Load word list into a set for fast lookup
word_list = set(words.words())
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Tokenizer: splits input into words
def tokenize(text):
    return re.findall(r"\b\w+\b|[^\w\s]", text)

# Check if a word is not in the dictionary
def is_misspelled(word):
    return word.lower() not in word_list

# Suggest closest words using Levenshtein-like fuzzy match
def suggest(word):
    suggestions = get_close_matches(word, word_list, n=5, cutoff=0.8)
    if suggestions:
        return max(suggestions, key=lambda w: word_frequency(w, 'en'))
    return word

def reduce_repeated_letters(word):
    """
    Iteratively reduce repeated letters in a word to try to find a valid dictionary match.
    This function reduces each group of repeated characters by one until a known word is found or no changes occur.
    """
    original = word
    candidate = word
    # Limit iterations to avoid infinite loops
    for _ in range(3):
        # For each group of repeated letters, reduce the count by one
        new_candidate = re.sub(r'(.)\1+', lambda m: m.group(1) * (len(m.group(0)) - 1), candidate)
        # If no change, break out of loop
        if new_candidate == candidate:
            break
        candidate = new_candidate
        if candidate.lower() in word_list:
            return candidate
    # If no valid word found, return the best candidate
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
            # Check that prediction is in dictionary and not the same word
            if predicted.lower() in word_list and predicted.lower() != original_word.lower():
                return predicted, score
    except:
        pass
    return original_word, 0

def correct_sentence(sentence):
    tokens = tokenize(sentence)
    corrected = []
    for i, token in enumerate(tokens):
        if re.match(r"\w+", token): 
            grammar_fix = apply_grammar_rules(token, i)
            if grammar_fix:
                if DEBUG:
                    print(f"[Rule] '{token}' → '{grammar_fix}' (grammar rule)")
                corrected_word = grammar_fix
            elif is_misspelled(token):
                reduced = reduce_repeated_letters(token)
                if reduced.lower() in word_list and reduced.lower() != token.lower():
                    if DEBUG:
                        print(f"[Reduce] '{token}' → '{reduced}' (repeated letters reduction)")
                    corrected_word = reduced
                else:
                    edit_suggestion = suggest(token)
                    original_freq = word_frequency(token, 'en')
                    edit_confidence = word_frequency(edit_suggestion, 'en') / (original_freq + 1e-6)

                    corrected_word = token 

                    if edit_suggestion != token and edit_confidence > 1.5:
                        if DEBUG:
                            print(f"[Edit] '{token}' → '{edit_suggestion}' (confidence: {edit_confidence:.2f})")
                        corrected_word = edit_suggestion
                    else:
                        swapped = list(token)
                        for i in range(len(swapped) - 1):
                            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                            candidate = "".join(swapped)
                            if candidate.lower() in word_list:
                                if DEBUG:
                                    print(f"[Swap] '{token}' → '{candidate}' (adjacent letter swap)")
                                corrected_word = candidate
                                break
                            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]

                        if corrected_word == token:
                            from itertools import permutations
                            perms = set("".join(p) for p in permutations(token) if len(p) == len(token))
                            for candidate in perms:
                                if candidate.lower() in word_list:
                                    if DEBUG:
                                        print(f"[Permute] '{token}' → '{candidate}' (letter rearrangement)")
                                    corrected_word = candidate
                                    break

                        if corrected_word == token:
                            context_word, context_score = context_suggest(sentence, i, token)
                            context_freq = word_frequency(context_word, 'en')
                            context_confidence = (context_freq / (original_freq + 1e-6)) * context_score
                            if (context_word != token 
                                and context_confidence > 1.0 
                                and similarity(token.lower(), context_word.lower()) > 0.5):
                                if DEBUG:
                                    print(f"[Context] '{token}' → '{context_word}' (confidence: {context_confidence:.2f})")
                                corrected_word = context_word
            else:
                corrected_word = token
            corrected.append(corrected_word.lower())
        else:
            corrected.append(token)

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

# 🔍 Test the spell checker
if __name__ == "__main__":
    print("Spell & Grammar Correction CLI. Type a sentence and press Enter (type 'exit' to quit):\n")
    while True:
        try:
            user_input = input("Input: ")
            if user_input.strip().lower() == "exit":
                break
            corrected = iterative_correct(user_input)
            print(f"Corrected: {corrected}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break