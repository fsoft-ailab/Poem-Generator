from typing import List

import torch


def get_bag_of_words_indices(file_path_bow, tokenizer) -> List[List[List[int]]]:
    bow_indices = []

    with open(file_path_bow, "r") as f:
        words = f.read().strip().split("\n")
    bow_indices.append(
        [tokenizer.encode(word.strip(), add_special_tokens=False)
         for word in words])
    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:

        num_words = len(single_bow)
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size + 2).to(device)
        for i in range(num_words):
            tokens = single_bow[i]
            one_hot_bow[i].scatter_(0, torch.tensor(tokens, device=device), 1)

        one_hot_bows_vectors.append(one_hot_bow)

    return one_hot_bows_vectors