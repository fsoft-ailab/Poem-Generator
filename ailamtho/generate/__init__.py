import torch
from transformers import RobertaTokenizer, GPT2Config, GPT2LMHeadModel, PhobertTokenizer
from underthesea import word_tokenize

from .generate_poem import generate_text
from .generate_topic import generate_text_pplm
from ailamtho.utils import Config, download, post_process
from ailamtho.utils import get_bag_of_words_indices, build_bows_one_hot_vectors


class PoemGenerator:
    """
    A simple generator class that generates poem with some words input

    Attributes
    ----------
    model_id : int
        0: Word-Level-GPT2Model
        1: Syllable-Level-GPT2Model
        2: Custom-loss-Model

    Methods
    -------
    generate_poem(context: str, n_stanzas)
        Generate poem with some words input
    """
    def __init__(self, model_id: int):
        """
        Parameters
        ----------
        model_id : int
            0: Word Level GPT2Model
            1: Syllable Level GPT2Model
            2: Our Custom Loss Model
        """

        if model_id not in [0, 1, 2]:
            raise ValueError('model id must be in [0, 1, 2]')

        self.model_id = model_id
        self.cfg = Config.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.model_id == 0:

            self.n_tokens_per_stanza = 30
            self.seg_word = True
            self.config_model = GPT2Config.from_json_file(download(self.cfg['word_level_gpt2model']['config']))
            merges_file = download(self.cfg['word_level_gpt2model']['tokenizer']['merges_file'])
            vocab_file = download(self.cfg['word_level_gpt2model']['tokenizer']['vocabs_file'])

            self.tokenizer = PhobertTokenizer(vocab_file, merges_file)
            self.tokenizer.add_tokens('\n')

            self.model = GPT2LMHeadModel(config=self.config_model)
            self.model.load_state_dict(torch.load(download(self.cfg['word_level_gpt2model']['weight'])))
            self.model.to(self.device)
            self.model.eval()

        elif self.model_id == 1:

            self.n_tokens_per_stanza = 40
            self.seg_word = False
            self.config_model = GPT2Config.from_json_file(download(self.cfg['syllable_level_gpt2model']['config']))
            merges_file = download(self.cfg['syllable_level_gpt2model']['tokenizer']['merges_file'])
            vocab_file = download(self.cfg['syllable_level_gpt2model']['tokenizer']['vocabs_file'])

            self.tokenizer = RobertaTokenizer(vocab_file, merges_file)
            self.tokenizer.add_tokens('\n')

            self.model = GPT2LMHeadModel(config=self.config_model)
            self.model.load_state_dict(torch.load(download(self.cfg['syllable_level_gpt2model']['weight'])))
            self.model.to(self.device)
            self.model.eval()

        else:

            self.n_tokens_per_stanza = 40
            self.seg_word = False
            merges_file = download(self.cfg['custom_loss_model']['tokenizer']['merges_file'])
            vocab_file = download(self.cfg['custom_loss_model']['tokenizer']['vocabs_file'])

            self.tokenizer = RobertaTokenizer(vocab_file, merges_file)
            self.tokenizer.add_tokens('\n')

            self.config_model = GPT2Config(vocab_size=self.tokenizer.vocab_size + 1, n_layer=6)

            self.model = GPT2LMHeadModel(config=self.config_model)
            self.model.load_state_dict(torch.load(download(self.cfg['custom_loss_model']['weight']))['state_dict'])
            self.model.to(self.device)
            self.model.eval()

    def generate_poem(self, context: str, n_stanzas=2):

        length = n_stanzas * self.n_tokens_per_stanza
        norm_text = context.strip('\n ').lower()
        if self.seg_word:
            norm_text = word_tokenize(norm_text, format='text')

        text: str = generate_text(self.model, self.tokenizer, context=norm_text, device=self.device, length=length,
                                  temperature=0.85, top_k=20, sample=True, show_time=False)
        poem = post_process(text, n_stanzas=n_stanzas)

        return poem


class ControlledPoemGenerator:
    """A class that generate poem towards the desired topic"""

    def __init__(self):

        self.id2topic = {0: 'gia_dinh', 1: 'tinh_yeu', 2: 'dich_benh', 3: 'que_huong', 4: 'le_tet'}
        self.cfg = Config.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config_model = GPT2Config.from_json_file(download(self.cfg['word_level_gpt2model']['config']))
        merges_file = download(self.cfg['word_level_gpt2model']['tokenizer']['merges_file'])
        vocab_file = download(self.cfg['word_level_gpt2model']['tokenizer']['vocabs_file'])

        self.tokenizer = PhobertTokenizer(vocab_file, merges_file)
        self.tokenizer.add_tokens('\n')

        self.model = GPT2LMHeadModel(config=self.config_model)
        self.model.load_state_dict(torch.load(download(self.cfg['word_level_gpt2model']['weight'])))
        self.model.to(self.device)
        self.model.eval()

        # Freeze layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Reconstruct one hot vector matrix of bows
        self.one_hot_bow_vector_list = []

        for idx in range(5):
            bow_ids = get_bag_of_words_indices(download(self.cfg['bow'][self.id2topic[idx]]), tokenizer=self.tokenizer)
            one_hot_bow_vector = build_bows_one_hot_vectors(bow_indices=bow_ids, device=self.device,
                                                            tokenizer=self.tokenizer)

            self.one_hot_bow_vector_list.append(one_hot_bow_vector)

    def generate_poem(self, context, topic_id: int, max_length=30):

        if topic_id not in [0, 1, 2, 3, 4]:
            raise ValueError('Topic Id must be in [0, 1, 2, 3, 4]')
        norm_text = context.strip('\n ').lower()
        norm_text = word_tokenize(norm_text, format='text')
        generated_text:str = generate_text_pplm(self.model, self.tokenizer, context=norm_text, device=self.device,
                                            one_hot_bows_vectors=self.one_hot_bow_vector_list[topic_id], length=max_length,
                                            loss_type=1, window_length=7, verbose=False, num_iterations=5,
                                            temperature=0.8, top_k=20)
        n_stanzas = generated_text.count('\n \n')
        poem = post_process(text=generated_text, n_stanzas=n_stanzas)

        return poem