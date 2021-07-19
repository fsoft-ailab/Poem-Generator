#### Table of contents
1. [Introduction](#introduction)
2. [Details](#details)
	- [Preprocess data](#processdata)
	- [Poem generation](#generate1)
		- [Word Level GPT2 Model](#wordlevel)
		- [Syllable Level GPT2 Model](#syllablelevel)
		- [Semantic Poem GPT2 Model](#customloss)
		- [Comparison](#comparison)
	- [Poem generation with desired topic](#generate2)
3. [QuickStart](#quickstart)
4. [Dataset](#dataset)
5. [Evaluation](#evaluation)

<p align="center">
  <h1 align="center", id="introduction">Vietnamese Poem Generator</h1>
</p>

In this project, we research on text generation to automatically generate
`luc-bat` genre poetry. We have experienced some [`GPT-2`](https://huggingface.co/transformers/model_doc/gpt2.html)
based models with different levels of Vietnamese language as syllable level or word level.
We also propose a new architecture built on top [`GPT-2`](https://huggingface.co/transformers/model_doc/gpt2.html)
model and addition loss to constant context through the entire poem. </br>

Our project can automatically generate a poem from the input of start words.
Besides, it can also automatically generate a poem with a particular topic. </br>

One of our challenges is the dataset problem.
So, we have to collect from many resources to create our dataset. [Details about our dataset](#dataset)


## Details <a name="details"></a>
### Preprocess data <a name="processdata"></a>
We normalize text to lower case and remove all special characters from raw poems.
Then, we split a poem into 4 verses blocks. If a poem or end block has lower
than 4 verses, we ignore it. Finally, we concatenate 8 blocks into one data
point. A tokenized data point corresponds to approximately 256 tokens
. We use `<pad>` token to padding in case of not enough 256 tokens.

### Poem generation from the input of start words<a name="generate1"></a>
We have experimented 3 models: Word Level GPT2 Model, Syllable GPT2 Model, Our
Custom Loss Model. The generated poem is evaluated base on three aspects: creativity
, score about grammar, semantic. We use automatic evaluation to evaluate the creativity
and grammar of the poem as well as model. About semantic of poems, we invite three
professional poets to assess semantic of poems in range 0-10 scores.

#### Word Level GPT2 Model (GPT2-WL) <a name="wordlevel"></a>
Before feeding text to tokenization process, we use [`underthesea`](https://github.com/undertheseanlp/underthesea)
frameworks to segment words. We train new [`fastBPE`](https://github.com/glample/fastBPE)
tokenizer to segment data points with subword units, using a vocabulary of 19795 subword types. <br/>

We use default `n_layer, n_head` of [OpenAI GPT2](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2config)
config and train from scratch with our `luc-bat` genre poetry dataset

```Note: You can download weigth of model from ```[`file`](https://github.com/mtb-hust/Poem-Generator/blob/master/ailamtho/config.yml)
#### Syllable Level GPT2 Model (GPT2-SL)<a name="syllablelevel"></a>
There are no word segmentation process in this experiment. Different from 
above model, we apply `Byte-Level BPE` tokenizer to segment data points with subword units,
using a vocabulary of 12860 subword types. <br/>

We also train from scratch with default `n_layer, n_head` of OpenAI GPT2 config.

```Note: You can download weigth of model from ```[`file`](https://github.com/mtb-hust/Poem-Generator/blob/master/ailamtho/config.yml)
#### Semantic Poem GPT2 Model (SP-GPT2)<a name="customloss"></a>
```Note: We will update details in the future```
#### Comparison <a name="comparison"></a>
| Model               | Creativity score (0-10) | Grammar score (0-100) | Human score (0-5) *(mean<img src="https://render.githubusercontent.com/render/math?math=\pm">std)* |
|---------------------|:-----------------------:|:---------------------:|:---------------------:|
| Word Level GPT2     |           9.55          |      84.26            |         3.02<img src="https://render.githubusercontent.com/render/math?math=\pm">1.49              |
| Syllable Level GPT2 |           9.64          |      84.54            |           `None`            |
| Semantic Poem GPT2  |           9.70           |      86.94            |         3.34<img src="https://render.githubusercontent.com/render/math?math=\pm">1.30              |


### Poem generation with desired topic <a name="generate2"></a>

Inspired by the method of [Plug and Play Language Models](https://arxiv.org/abs/1912.02164) paper,
We build 5 bags of word topics: `gia-dinh`, `tinh-yeu`, `dich-benh`, `que-huong`, `tinh-yeu`
and use our model to generate a poem with the desired topic

## Quickstart <a name="quickstart"></a>
Please click the image below to know how to generate poems </br>

<a href="https://colab.research.google.com/drive/1gvHp-_ZC4twOxz6b9P9UTx1-n0AZlqG9" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Dataset <a name="dataset"></a>
We collect poems from many resources as facebook groups, [Vườn Thơ Tkaraoke](https://poem.tkaraoke.com/),
[lucbat.com](http://lucbat.com/),... Our dataset contains **171188** poems with different genres:
`luc-bat`, `5-chu`, `7-chu`, `8-chu`, `4-chu`. Detail in table below: <br/>

| Genre   | n_poems |
|---------|:-------:|
| `luc-bat` |  87609  |
| `5-chu`   |   6747  |
| `7-chu`   |  41286  |
| `8-chu`   |  34440  |
| `4-chu`   |   1106  |

You can download our dataset [here](https://drive.google.com/file/d/1q4AicnHAD8kxpEPihfbFDaG-KR_CSGQq/view?usp=sharing)
## Evaluation <a name="evaluation"></a>

### Rule

Vietnamese Poems have some explicit rules related to rhymes and tones for each stanza.


#### Rhyme check

A stanza rhyme form:

    u    u    u     u   u   R1
    u    u    u     u   u   R1    u   R2
    u    u    u     u   u   R2
    u    u    u     u   u   R2    u   R3
    . . . 

    u: undefined
    Rx: rhyme need checked
    
    As you can see, the form requires rhyme in position marked as the same notation(R) to be the same.
    For more information, 
      the first pair of sentences, 
                the sixth words of the first sentence(R1) - the sixth words of the next sentence(R1)
      from the sencond sentence,  
                the eighth words(R2) - the following sixth words(R2)
                the eighth words(R2) - the next sixth words(R2)
                              
#### Tone check

A stanza tone form:

    u    B    u     T   u   B
    u    B    u     T   u   B    u   B
    u    B    u     T   u   B
    u    B    u     T   u   B    u   B
    . . . 

    B: Even tone
    T: Uneven tone

    Follow the above form, you can know clearly how tone check works
### Scoring

Each stanza with n pair of sentences has: (3*n – 1) words to check rhyme and (7*n) words to check tone.

    **TOTAL_SCORE  = 100 - 70*WRONG_RHYMES_RATE - 30*WRONG_TONES_RATE**
        WRONG_RHYMES_RATE = WRONG_RHYMES_COUNT/(3*n – 1)
        WRONG_TONES_RATE = WRONG_TONES_COUNT/(7*N)
	
### Example usage
```python 
from src.utils.check_rule import *
print(check_rule(input))
# errors check, marked poem, length, tone and rhyme errors returned
print(calculate_score(input))
# Score returned
```

## Contact

- Supervisor: [Tuan Nguyen](https://www.facebook.com/nttuan8) 
- Team Members: [Hanh Pham](https://www.facebook.com/phamvanhanh2000), [Manh Truong](https://www.facebook.com/mtbhust/), [Hoang Duc](https://www.facebook.com/duc.once), [Phuc Tan](https://www.facebook.com/profile.php?id=100006032584238)
## Sponsor
   Special thanks to [FPT Software AI Lab](https://ai.fpt-software.com) for sponsoring this project

## License
	MIT License

	Copyright (c) 2021 FPT Software AI Lab

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
