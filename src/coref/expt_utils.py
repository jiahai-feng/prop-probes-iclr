import einops
import torch
def generate_slices(start, end, batch_size):
    for s in range(start, end, batch_size):
        yield slice(s, min(end, s + batch_size))

def lookup_answer_logits(logits, answer_tokens, normalize=True, query_position=None):
    """
    logits: (batch_size, seq_len, vocab_size)
    answer_tokens: (batch_size, num_answers)
    query_position: None | (batch_size)

    """
    # Only the final logits are relevant for the answer
    device = logits.device
    if query_position is None:
        final_logits = logits[:, -1, :]
    else:
        final_logits = logits.gather(
            dim=1,
            index=einops.repeat(
                query_position, "batch -> batch dummy dim", dummy=1, dim=logits.shape[2]
            ).to(device),
        )[:, 0, :]
        assert final_logits.shape == logits[:, -1, :].shape
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens.to(device))
    if normalize:
        answer_logits = answer_logits - final_logits.logsumexp(dim=-1, keepdim=True)
    return answer_logits


def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    # Only the final logits are relevant for the answer
    # will take [:, 0] - [:, 1]
    answer_logits = lookup_answer_logits(
        logits.cpu(), answer_tokens.cpu(), normalize=False
    )
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()


from rich import print as rprint


def pretty_print_logits(tokenizer, logits):
    top_k = 10
    token_probs = logits.softmax(-1)
    sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)
    # String formatting syntax - the first number gives the number of characters to pad to, the second number gives the number of decimal places.
    # rprint gives rich text printing
    for i in range(top_k):
        print(
            f"Top {i}th token. Logit: {logits[sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{tokenizer.decode(sorted_token_values[i])}|"
        )


def repeat(num, fn):
    return torch.stack([fn(i) for i in range(num)])


def reduce_logit_mean(all_logits):
    """
    all_logits: [..., name, batch, answer]
    """
    return all_logits.mean(dim=-2)


def reduce_acc(all_logits, intended_answers):
    """
    all_logits: [..., name, batch, answer]
    intended_answers: [..., name]
    """

    intended_answers = torch.tensor(intended_answers)
    return (
        (all_logits.argmax(dim=-1) == intended_answers[..., None]).float().mean(dim=-1)
    )


def reduce_mean_correct_logit(all_logits, intended_answers, ignore_null=False):
    # intended_answers: [..., name]
    if ignore_null:
        all_logits = all_logits[..., :-1, :, :]
    mat = reduce_logit_mean(all_logits)  # [..., name, answer]
    intended_answers = torch.tensor(intended_answers)[..., None]
    return torch.gather(mat, dim=-1, index=intended_answers)[..., 0]


def reduce_balanced_acc(all_logits, intended_answers, offsets=None, ignore_null=False):
    if ignore_null:
        all_logits = all_logits[..., :-1, :, :]
    if offsets is None:
        # median is taken over name and batch
        median = torch.flatten(all_logits, start_dim=-3, end_dim=-2).quantile(
            dim=-2, q=0.5
        )  # we've lost both dimensions
        offsets = median[..., None, None, :]  # restore dimensions
    rebalanced_logits = all_logits - offsets
    return reduce_acc(rebalanced_logits, intended_answers)


def reduce_distractor_acc(tensor, intended_answers):
    """
    tensor: [..., name, batch, attr]
    """
    baseline = tensor[..., -1:, :, :].quantile(dim=-2, q=0.5, keepdim=True)
    mat = tensor[..., :-1, :, :]
    return reduce_balanced_acc(mat, intended_answers=intended_answers, offsets=baseline)
