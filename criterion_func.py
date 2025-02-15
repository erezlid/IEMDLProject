from torch.nn import functional as nnf

def criterion_func(proxy_loss, prefix, tokens, logits, alpha=0.1):
    p_loss = proxy_loss(prefix)
    cross_loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
    final_loss = cross_loss + alpha * p_loss
    return final_loss