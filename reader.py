import collections
import logging
import os
import sys
import random
import string
import unicodedata

import numpy as np
from lxml import etree
import torch
import torch.nn as nn
from torch.autograd import Variable

import util

log = logging.getLogger(__name__)

# create a file handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

punctuation = string.punctuation + '’”…—“'

formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.INFO)


MAX_ATTEMPTS = 500  # Short circuit at this point if we haven't finished a sentence by then
MAX_DUPLICATES = 1  # Maximum number of times we're willing to see the same word
MIN_WORDS = 4  # Min number of words we'd like to see or re-roll
MIN_MATCH_WORDS = 4  # We insist on seeing at least this many words that match our letter
CANDIDATE_SETS = 20  # Number of different sets we generate before averaging


# Stupid nonwords it insists on trying
WORD_BLACKLIST = set(['b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                      'ed', 'ing', 'er', 'en', 'ies', 'th', 'est', 'es', 'ee'])

# From NLTK
WHITELIST = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'us', 'am',
                 'xylophone', 'xxx', 'xray', 'xactly'])



def exclude_word(rnn, word, sentence, word_set, revmap):
    """Determine if a particular word should be dropped"""
    word = word.strip(punctuation).lower().strip()  # Normalize with no punctuation & lowercase only
    sentence = [w.strip(punctuation).lower() for w in sentence]

    # Reject nonwords
    if word not in word_set:
        return True

    # Never double any word
    if len(sentence) > 0 and word == sentence[-1]:
        return True

    # Reject any number of words over our MAX_DUPLICATES threshold, unless they're in the WHITELIST list
    if word not in WHITELIST and sentence.count(word) >= MAX_DUPLICATES:
        return True

    # And even then, don't let us repeat WHITELIST more than a few times
    if sentence.count(word) >= MAX_DUPLICATES * 2:
        return True

    # Reject any words in our stupid blacklist
    if word in WORD_BLACKLIST:
        return True

    # Accept any words in the WHITELIST list
    if word in WHITELIST:
        return False

    # Finally, reject any words that are too improbable unless it's X because sigh
    if len(word) > 1:
        prob = calc_word_prob(rnn, word, revmap)
        threshold = threshold_by_length(word)
        #log.info("%s: %s len: %d prob: %.4f threshold: %.4f", "WORD" if prob >= threshold else "NOT", word, len(word), prob, threshold)
        if prob < threshold:
            #log.info("%s is NOT a word prob=%.4f (thres=%.2f)?? [%s]", word, prob, threshold, " ".join(sentence))
            return True
    return False


def threshold_by_length(word):
    """Get a pre-determined probability threshold based on heuristics"""
    threshold = -5
    if len(word) > 6:
        threshold = -20
    if len(word) == 6:
        threshold = -18
    if len(word) == 5:
        threshold = -16
    if len(word) == 4:
        threshold = -14
    if len(word) == 3:
        threshold = -13
    # If it's a rare starting letter, lower this
    if word.startswith('x') or word.startswith('z'):
        threshold -= 10
    return threshold


def generate_with_coersion(rnn, charmap, revmap,
                           hidden=None,
                           temperature=0.3,
                           letter_preference='k',
                           sample_from=10,
                           word_set=None):
    """Try to pick words with specific letters by sampling from `sample_from` most plausible letters"""
    # log.info("Letter %s: sample from: %d, temp=%.4f", letter_preference, sample_from, temperature)
    rnn.eval()
    if hidden:  # Prime with some context if we didn't pass a hidden state
        start = ", "
    else:
        start = "It was springtime in America and "
        hidden = Variable(torch.zeros(rnn.n_layers, 1, rnn.hidden_size).cuda())

    start += letter_preference.strip()  # This will be last (or current) character we predicted
    inp = Variable(torch.LongTensor([util.doc_to_seq(start, revmap)]), volatile=True).cuda()

    logits, hidden = rnn(inp, hidden, seq_lengths=[len(start)])

    sentence = []  # The sentence we've built
    current_word = [letter_preference]  # The current word we're constructing
    predicted_char = ""
    count = 0

    # Get our preferred letter index (either case)
    target_i = (revmap[letter_preference.lower()],
                revmap[letter_preference.upper()])

    while count < MAX_ATTEMPTS:
        lengths = len(start) if count == 0 else 1
        logits, hidden = rnn(inp, hidden, seq_lengths=[lengths])
        logits = logits[-1, :]
        output_dist = logits.data.div(temperature).exp()
        choice = None

        # # If we're starting a new word, try to get one with our preferred letter
        if predicted_char == " ":
            if letter_preference == 'a':
                sample_from = 3  # Very small for super-common letters
            top = torch.multinomial(output_dist, sample_from, replacement=False)
            for t in top:
                if t in target_i:
                    choice = t
                    break
        # Otherwise just pick the absolute best if we didn't find our preference or are mid-word
        if not choice:
            choice = torch.multinomial(output_dist, 1)[0]
        predicted_char = charmap[choice]

        word = "".join(current_word)

        # When the word is completed, check if it meets any of our exclusionary criteria
        if predicted_char == " ":
            if len(word) > 1:
                if exclude_word(rnn, word, sentence, word_set, revmap):
                    # Erase the in-progress word and re-roll
                    current_word = [] if len(sentence) > 0 else [letter_preference]
                    continue

                sentence.append(word.strip())
                current_word = []

        elif predicted_char == ".":
            # Last check
            if exclude_word(rnn, word, sentence, word_set, revmap):
                 current_word = []
                 continue

            # We're done, congrats
            word = word.strip() + "."
            sentence.append(word)
            break

        # We're still mid-word, so just append this character
        else:
            current_word.append(predicted_char)

        # Set up the next input state with this character
        inp = Variable(torch.LongTensor([util.doc_to_seq(predicted_char, revmap)]), volatile=True).cuda()
        count += 1

    starts = [w[0].lower() for w in sentence if w[0].lower() == letter_preference.lower()]
    min_match = 1 if letter_preference in ('x', 'z') else MIN_MATCH_WORDS

    if (len(sentence) < MIN_WORDS or len(starts) < min_match):
         temperature += 0.05
         sample_from = min(sample_from + 2, 150)
         sentence, hidden = generate_with_coersion(rnn,
                                           charmap, revmap,
                                           temperature=temperature,
                                           sample_from=sample_from,
                                           letter_preference=letter_preference,
                                           word_set=word_set)
    return sentence, hidden



def generate(rnn, charmap, revmap, predict_len=1000, prime_str="In the woods, ", temperature=0.8):
    """Generate unrestricted text"""
    rnn.eval()
    inp = Variable(torch.LongTensor([util.doc_to_seq(prime_str, revmap)]), volatile=True).cuda()
    hidden = Variable(torch.zeros(rnn.n_layers, 1, rnn.hidden_size).cuda())
    logits, hidden = rnn(inp, hidden, seq_lengths=[len(prime_str)])
    predicted = ""
    for p in range(predict_len):
        logits, hidden = rnn(inp, hidden, seq_lengths=[1])
        logits = logits[-1, :]
        output_dist = logits.data.div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]  # Get the value from the tensor
        predicted_char = charmap[top_i]
        predicted += predicted_char
        inp = Variable(torch.LongTensor([util.doc_to_seq(predicted_char, revmap)]), volatile=True).cuda()
    return predicted


def calc_word_prob(rnn, word, revmap, hidden=None, average=False):
    """Given some hidden state, calculate the probabilty of a given word"""
    word_buffer = " "
    word += ' '
    probs = []
    if not hidden:
        hidden = Variable(torch.zeros(rnn.n_layers, 1, rnn.hidden_size).cuda())
    for i, c in enumerate(word):
        inp = Variable(torch.LongTensor([util.doc_to_seq(word_buffer, revmap)]), volatile=True).cuda()
        logits, hidden = rnn(inp, hidden, seq_lengths=[len(word_buffer)])
        # Get the probability of the i + 1th letter
        m = nn.LogSoftmax()
        all_probs = m(logits)[-1, :]
        index = revmap[c]
        prob = all_probs[index].data[0]
        probs.append(prob)
        word_buffer += c
    probs = np.sum(np.array(probs))
    if average:
        # Return an average by length of string
        return probs / len(word)
    return probs


def run():
    words = util.load_data()
    charmap, revmap = util.map_all_chars(words)
    n_characters = len(charmap.keys())


    counts = collections.Counter([word.strip(punctuation).lower().strip() for word in words.split()])

    word_set = set()
    for word in counts.most_common():
        if word[1] == 5:
            break
        word_set.add(word[0])

    log.info("Running with a vocab size of %d characters, %d total words, %d unique words", n_characters, len(words), len(word_set))
    rnn = torch.load(util.MODEL_NAME)
    rnn.cuda()

    hidden = None
    sets = {}
    for i in range(CANDIDATE_SETS):
        for k in string.ascii_lowercase:
            sets[k] = []
            sent, hidden = generate_with_coersion(rnn, charmap, revmap,
                                                   hidden=hidden,
                                                   letter_preference=k, word_set=word_set)
            sent = " ".join(sent).capitalize()
            print(sent)
            sets[k].append(sent)
    #
    # for k in string.ascii_lowercase:
    #      best_sent = sets[k][0]
    #      max_prob = calc_word_prob(rnn, best_sent, revmap)
    #      for sent in sets[k]:
    #          prob = calc_word_prob(rnn, sent, revmap, average=True)
    #          if prob > max_prob:
    #              max_prob = prob
    #              best_sent = sent
    #
    #      print(best_sent.replace(' .', '.'))  # Patch up the final period if it was joined funny


if __name__ == '__main__':
    run()
