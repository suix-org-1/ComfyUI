import torch
import importlib.metadata

# 检查transformers库版本
try:
    transformers_version = importlib.metadata.version('transformers')
    major, minor = map(int, transformers_version.split('.')[:2])
    use_new_api = (major > 4) or (major == 4 and minor >= 49)
except (importlib.metadata.PackageNotFoundError, ValueError):
    # 如果无法确定版本，假设使用旧版本API
    use_new_api = False

# 根据版本选择正确的导入
if use_new_api:
    try:
        # 在新版本中，LogitsWarper已合并到LogitsProcessor
        from transformers import LogitsProcessor as BaseClass
        print("[IndexTTS] 使用transformers新版API (>= 4.49)，LogitsProcessor")
    except ImportError:
        # 如果新导入失败，尝试旧版本
        from transformers import LogitsWarper as BaseClass
        print("[IndexTTS] 使用transformers旧版API (< 4.49)，LogitsWarper")
else:
    # 旧版本继续使用LogitsWarper
    try:
        from transformers import LogitsWarper as BaseClass
        print("[IndexTTS] 使用transformers旧版API (< 4.49)，LogitsWarper")
    except ImportError:
        # 如果旧导入失败，尝试新版本
        from transformers import LogitsProcessor as BaseClass
        print("[IndexTTS] 使用transformers新版API (>= 4.49)，LogitsProcessor")


class TypicalLogitsWarper(BaseClass):
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
