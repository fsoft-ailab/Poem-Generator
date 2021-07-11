from typing import List


def post_process(text: str, n_stanzas=2):

    text = text.replace('_', ' ')
    text = text.replace('@@', '')
    text = text.strip('<s>')

    lines = text.split('\n')
    lines: List[str] = [l.strip(' ') for l in lines]
    lines = [l.capitalize() for l in lines]
    text = '\n'.join(lines)
    stanzas = text.split('\n\n')  # list poems

    poem = '\n\n'.join(stanzas[:n_stanzas])

    return poem