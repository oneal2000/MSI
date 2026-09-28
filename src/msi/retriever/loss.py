"""Global-batch multi-positive MNR loss with differentiable DDP gathering."""
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def ranking_loss(query, candidates, candidate_ids, positive_mask, temperature=0.05):
    scores = query.float() @ candidates.float().T / temperature
    # Repeated skill documents represent one fact, even when they occur in
    # several triplets. Collapse duplicate columns before the softmax.
    seen, keep = set(), []
    for idx, sid in enumerate(candidate_ids.tolist()):
        if sid not in seen:
            keep.append(idx)
            seen.add(sid)
    scores = scores[:, keep]
    positives = positive_mask[:, candidate_ids[keep]]
    if not positives.any(dim=1).all():
        raise ValueError("query has no positive candidate")
    return (torch.logsumexp(scores, dim=1)
            - torch.logsumexp(scores.masked_fill(~positives, -torch.inf), dim=1)).mean()


def distributed_loss(query, candidates, candidate_ids, positive_mask, temperature=0.05):
    if dist.is_initialized() and dist.get_world_size() > 1:
        candidates = torch.cat(all_gather(candidates), dim=0)
        ids = [torch.empty_like(candidate_ids) for _ in range(dist.get_world_size())]
        dist.all_gather(ids, candidate_ids)
        candidate_ids = torch.cat(ids)
    return ranking_loss(query, candidates, candidate_ids, positive_mask, temperature)
