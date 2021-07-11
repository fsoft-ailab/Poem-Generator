from operator import add

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import trange
from torch.autograd import Variable

from ..utils import top_k_filter


PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def perturb_past(past, model, last, unpert_past=None, unpert_logits=None, accumulated_hidden=None,
                 grad_norms=None, stepsize=0.01, one_hot_bows_vectors=None, loss_type=0, num_iterations=3,
                 horizon_length=1, window_length=0, decay=False, gamma=1.5, kl_scale=0.01, device='cuda', verbose=True):
    # Generate inital perturbed past
    past = list(past)
    for i in range(len(past)):
        temp = torch.cat(past[i], dim=0)
        past[i] = torch.unsqueeze(temp, 1)
    past = tuple(past)

    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length > 0:
        ones_key_val_shape = (
            tuple(past[0].shape[:-2])
            + tuple([window_length])
            + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
            tuple(past[0].shape[:-2])
            + tuple([curr_length - window_length])
            + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbose:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape

        new_pertubed_past = []
        for i in range(len(perturbed_past)):
            key_value = (perturbed_past[i][0], perturbed_past[i][1])
            new_pertubed_past.append(key_value)
        new_pertubed_past = tuple(new_pertubed_past)

        output = model(input_ids=last, past_key_values=new_pertubed_past, return_dict=True, output_hidden_states=True)
        all_logits, all_hidden = output.logits, output.hidden_states

        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                loss += bow_loss
                loss_list.append(bow_loss)
            if verbose:
                print(" pplm_bow_loss:", loss.data.cpu().numpy())

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                unpert_probs + SMALL_CONST *
                (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbose:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbose:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def generate_text_pplm(model, tokenizer, context=None, past=None, device="cuda", perturb=True, one_hot_bows_vectors=None,
                       loss_type=1, length=100, stepsize=0.02, temperature=1, top_k=10, sample=True,
                       num_iterations=3, grad_length=10000, horizon_length=1, window_length=5, decay=False,
                       gamma=1.5, gm_scale=0.9, kl_scale=0.01, verbose=False):

    output_so_far = None

    # Tokenize
    tokens_id = tokenizer(context)['input_ids'][:-1]
    context_t = torch.tensor(tokens_id, device=device, dtype=torch.long)

    while len(context_t.shape) < 2:
        context_t = context_t.unsqueeze(0)
    output_so_far = context_t

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    if verbose:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)
    for i in range_func:

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                output = model(output_so_far[:, :-1], return_dict=True, output_hidden_states=True)
                past = output.past_key_values

        output = model(output_so_far, return_dict=True, output_hidden_states=True)
        unpert_logits, unpert_past, unpert_all_hidden = output.logits, output.past_key_values, output.hidden_states
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbose=verbose
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        output = model(last, past_key_values=pert_past, return_dict=True, output_hidden_states=True)
        pert_logits, past, pert_all_hidden = output.logits, output.past_key_values, output.hidden_states

        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
    if verbose:
        print(tokenizer.decode(output_so_far.tolist()[0]))

    generated_text = tokenizer.decode(output_so_far.tolist()[0])

    return generated_text