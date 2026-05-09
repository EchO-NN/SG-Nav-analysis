import torch
import torch.nn.functional as F


def soft_cross_entropy(logits, target_probs):
    log_probs = torch.log_softmax(logits, dim=0)
    return -(target_probs * log_probs).sum()


def distillation_kl(student_logits, teacher_scores, temperature=2.0):
    teacher_probs = torch.softmax(teacher_scores / temperature, dim=0)
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=0)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return loss * temperature * temperature


def pairwise_ranking_loss(logits, costs, margin=0.1, max_pairs=64):
    valid = torch.isfinite(costs)
    logits = logits[valid]
    costs = costs[valid]
    if logits.numel() <= 1:
        return logits.sum() * 0.0

    best = torch.argmin(costs)
    s_pos = logits[best]
    neg = torch.arange(logits.numel(), device=logits.device)
    neg = neg[neg != best]
    if neg.numel() > max_pairs:
        perm = torch.randperm(neg.numel(), device=logits.device)[:max_pairs]
        neg = neg[perm]
    return torch.relu(float(margin) - s_pos + logits[neg]).mean()

