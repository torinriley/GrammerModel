import re
from itertools import permutations
from nltk.corpus import words
from difflib import get_close_matches, SequenceMatcher
from wordfreq import word_frequency
from transformers import pipeline

DEBUG = True
word_list = set(words.words())
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

def tokenize(text):
    return re.findall(r"\b\w+\b|[^\w\s]", text)

def is_misspelled(word):
    return word.lower() not in word_list

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

def suggest(word):
    suggestions = get_close_matches(word, word_list, n=5, cutoff=0.8)
    if suggestions:
        return max(suggestions, key=lambda w: word_frequency(w, 'en'))
    return word

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

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


def correctness_pipeline(sentence):
    tokens = tokenize(sentence)
    corrected = []
    for i, token in enumerate(tokens):
        if re.match(r"\w+", token): 
            if is_misspelled(token):
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

    return corrected