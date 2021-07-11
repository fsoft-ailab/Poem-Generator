import time

import torch
import torch.nn.functional as F

from ailamtho.utils import top_k_filter


def generate_text(model, tokenizer, context, device, length=100, temperature=1.0, top_k=20, sample=True,
                  show_time=True):

    output_so_far = None

    # Tokenize
    tokens_id = tokenizer(context)['input_ids'][:-1]
    context_t = torch.tensor(tokens_id, device=device, dtype=torch.long)

    while len(context_t.shape) < 2:
        context_t = context_t.unsqueeze(0)
    output_so_far = context_t

    last = None
    start = time.time()
    for i in range(length):

        if last is None:
            output = model(output_so_far, return_dict=True)
        else:
            output = model(input_ids=last, past_key_values=past_key_values, return_dict=True)

        logits, past_key_values = output.logits, output.past_key_values
        logits = logits[:, -1, :] / temperature

        # Top-K
        pert_logits = top_k_filter(logits, k=top_k)
        probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(probs, num_samples=1)
        else:
            _, last = torch.topk(probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )

    if show_time:
        print('Total time: {:.4f}'.format(time.time() - start))

    return tokenizer.decode(output_so_far.tolist()[0])





