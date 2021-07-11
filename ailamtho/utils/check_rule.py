# -*- coding: utf-8 -*-
"""check_rhyme_final.ipynb

Scripts to evaluate Six-Eight poems

Author: GenAI team: TruongBM1

Date: 11/04/2021
"""

import ast
from math import ceil, floor

try:
    from importlib import resources
except ImportError:
    import importlib_resources as resources


def load_data(filename: str):

    with resources.open_text('ailamtho.resources', filename) as file:
        text = file.read()

    content = ast.literal_eval(text)
    return content


vowels_path = "start_vowels.txt"
start_vowels = load_data(vowels_path)

huyen = start_vowels['huyen']
sac = start_vowels['sac']
nang = start_vowels['nang']
hoi = start_vowels['hoi']
nga = start_vowels['nga']
khong_dau = start_vowels['khong_dau']

list_start_vowels = []
list_start_vowels.extend(huyen)
list_start_vowels.extend(sac)
list_start_vowels.extend(nang)
list_start_vowels.extend(hoi)
list_start_vowels.extend(nga)
list_start_vowels.extend(khong_dau)

rhyme_path = "rhymes.txt"

rhymes_dict = load_data(rhyme_path)


even_chars = []

even_chars.extend(huyen)
even_chars.extend(khong_dau)

tone_dict = load_data("tone_dict.txt")


def is_stanza(sentences: str):
    """
      Check if input is a stanza or not

      param sentences: sentences to check

      return: is stanza or not
    """
    return len(sentences.split("\n\n")) == 1


def split_word(word):
    """
        Split word by 2 part, starting and ending

        param word: word to split

        return: ending part of word
        Ex: mùa -> ùa
    """
    word_length = len(word)
    start_index = 0
    prev = ''
    for i in range(word_length):
        if prev == 'g' and word[i] == 'i':
            continue
        if prev == 'q' and word[i] == 'u':
            continue
        if word[i] in list_start_vowels:
            start_index = i
            break
        prev = word[i]
    return word[start_index:]


def compare(word1: str, word2: str):
    """
      Check 2 words rhyme if the same

      param word1, word2: words to check

      return: is the same rhyme or not
    """
    rhyme1 = split_word(word1)
    rhyme2 = split_word(word2)

    if rhyme2 in rhymes_dict[rhyme1]:
        return True
    return False


def check_rhyme_pair(prev_sentence: str, cur_sentence: str, prev_eight_words_rhyme=""):
    """
        Check 2 words rhyme if the same

        param word1, word2: words to check

        return: is the same rhyme or not
      """
    rhyme_errors = 0
    length_errors = 0

    prev_length = len(prev_sentence.split(" "))
    cur_length = len(cur_sentence.split(" "))

    if prev_length != 6:
        prev_sentence = "(L)" + prev_sentence
        length_errors = length_errors + 1
        print(1)

    if cur_length != 8:
        cur_sentence = "(L)" + cur_sentence
        length_errors = length_errors + 1

    prev_words = prev_sentence.split(" ")
    cur_words = cur_sentence.split(" ")

    if prev_eight_words_rhyme == "":
        try:
            if not compare(prev_words[5], cur_words[5]):
                cur_words[5] = cur_words[5] + "(V)"
                rhyme_errors = rhyme_errors + 1
        except Exception as e:
            print(f"{e} + {cur_sentence}")
            pass
    if prev_eight_words_rhyme != "":
        try:
            if not compare(prev_words[5], prev_eight_words_rhyme):
                prev_words[5] = prev_words[5] + "(V)"
                rhyme_errors = rhyme_errors + 1
        except Exception as e:
            print(f"{e} + {cur_sentence}")
            pass
        try:
            if not compare(prev_eight_words_rhyme, cur_words[5]):
                cur_words[5] = cur_words[5] + "(V)"
                rhyme_errors = rhyme_errors + 1
        except Exception as e:
            print(f"{e} + {cur_sentence}")
            pass
    prev_sentence = " ".join(prev_words)
    cur_sentence = " ".join(cur_words)

    return prev_sentence, cur_sentence, cur_words[-1], rhyme_errors, length_errors


def check_rhyme_stanza(stanza: str):
    """
        Check rhyme by stanza

        param stanza: input stanza to check

        return: res: stanza after check filter and error highlighted
                total_rhyme_errors: total rhyme errors
                total_length_errors: total length errors
      """
    sentences = stanza.split("\n")
    first_words = sentences[0].split(" ")
    start_index = 0
    prev_eight_words_rhyme = ""
    total_rhyme_errors = 0
    total_length_errors = 0

    if len(first_words) == 8:
        prev_eight_words_rhyme = split_word(first_words[7])
        start_index = 1

    for i in range(start_index, len(sentences), 2):
        if i+1 == len(sentences):
            sentences.append("Missing ending sentence")
        sentences[i], sentences[i+1], prev_eight_words_rhyme, rhyme_errors, length_errors =\
            check_rhyme_pair(sentences[i], sentences[i + 1], prev_eight_words_rhyme)
        total_rhyme_errors = total_rhyme_errors + rhyme_errors
        total_length_errors = total_length_errors + length_errors
    res = "\n".join(sentences)
    return res, total_rhyme_errors, total_length_errors


def get_tone(word: str):
    """
          Check word is even tone or not

          param word: word to check tone

          return: even or uneven
        """
    first_char = split_word(word)
    first_char = first_char[0]
    for i in even_chars:
        if first_char == i:
            return 'even'
    try:
        second_char = first_char[1]
        for i in even_chars:
            if second_char == i:
                return 'even'
    except:
        pass
    return 'uneven'


def check_tone_sentence(sentence: str):
    """
        Check sentence is on the right form of even or uneven rule

        param sentence: sentence to check tone

        return: sentences after added notation to highlight error
                total_wrong_tone: total wrong tone in sentence
      """
    words = sentence.split(" ")
    length = len(words)
    if length != 6 and length != 8:
        return "(L)"+sentence, 0
    cur_tone_dict = tone_dict[length]
    total_wrong_tone = 0
    for i in cur_tone_dict:
        if get_tone(words[i]) != cur_tone_dict[i]:
            total_wrong_tone = total_wrong_tone + 1
            words[i] = words[i] + "(T)"
    return " ".join(words), total_wrong_tone


def check_tone_stanza(stanza: str):
    """
        Check stanza is on the right form of even or uneven rule

        param sentence: stanza to check tone

        return: stanza after added notation to highlight error
                total_wrong_tone: total wrong tone in sentence
      """
    sentences = stanza.split("\n")
    total_wrong = 0
    for i in range(len(sentences)):
        current_wrong = 0
        sentences[i], current_wrong = check_tone_sentence(sentences[i])
        total_wrong = total_wrong + current_wrong
    return "\n".join(sentences), total_wrong


def preprocess_stanza(stanza: str):
    """
       A function to process Stanza to remove all unnecessary blank

       param sentence: stanza to process

       return: stanza processed
     """
    sentences = stanza.split("\n")
    sentences_out = []
    for sentence in sentences:
        words = sentence.split(" ")
        words_out = []
        for word in words:
            if word:
                words_out.append(word)
        sentences_out.append(" ".join(words_out))
    return "\n".join(sentences_out)


def check_rule(stanza: str):
    """
      A function to check both rhyme and tone rule

      param sentence: stanza to check

      return: stanza processed
    """
    if not is_stanza(stanza):
        print(stanza + ": is not a stanza")
        return
    stanza = preprocess_stanza(stanza)
    stanza, total_rhyme_errors, total_length_errors = check_rhyme_stanza(stanza)
    stanza, total_wrong_tone = check_tone_stanza(stanza)
    return stanza, total_length_errors, total_rhyme_errors, total_wrong_tone


def calculate_score_by_error(stanza_length: int, total_length_errors=0, total_rhyme_errors=0, total_wrong_tone=0):
    """
      A function to calculate score for the Stanza by length, rhyme and tone errors
          Currently doesnt punish the length error

      param sentence: stanza_length,
                      total_length_errors,
                      total_rhyme_errors,
                      total_wrong_tone

      return: score calculated by formula that rhyme accounts for 70% score rate and 30% left for tone
    """

    num_six = ceil(stanza_length/2)
    num_eight = floor(stanza_length/2)

    rhyme_minus_points = 70*total_rhyme_errors/(num_six + 2*num_eight-1)
    tone_minus_points = 30*total_wrong_tone/(3*num_six+4*num_eight)

    return 100 - rhyme_minus_points - tone_minus_points


def calculate_stanza_score(stanza: str):
    """
       A function to calculate score for the Stanza

       param sentence: stanza

       return: score  after checked by rule and calculated by formula that rhyme accounts for 70% score rate
        and 30% left for tone
     """

    stanza = preprocess_stanza(stanza)
    length = len(stanza.split("\n"))

    try:
        stanza, total_length_errors, total_rhyme_errors, total_wrong_tone = check_rule(stanza)

        score = calculate_score_by_error(length, total_length_errors, total_rhyme_errors, total_wrong_tone)
    except Exception as e:
        print(e)
        score = 0
    return score


def calculate_score(poem: str):
    """
       A function to calculate score for a poem that may have some stanzas

       param sentence: poem

       return: score  after checked by rule and calculated by formula that rhyme accounts for 70% score rate
       and 30% left for tone
     """
    sum_ = 0
    count = 0
    for i in poem.split("\n\n"):
        count += 1
        sum_ = sum_ + calculate_stanza_score(i)
    return sum_/count
