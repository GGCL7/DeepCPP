from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModel


ESM_DIR: str = "ESM_pre_model"

__all__ = [
    "ESM_DIR",
    "get_tokenizer_and_model",
    "masked_mean_max",
    "extract_features_concat_mean_max",
    "esmfeature",
]

def get_tokenizer_and_model(
    esm_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if esm_dir is None:
        esm_dir = ESM_DIR
    tokenizer = AutoTokenizer.from_pretrained(esm_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(esm_dir, trust_remote_code=True).to(device)
    model.eval()
    return tokenizer, model, device

def masked_mean_max(
    hidden: torch.Tensor,         # [B, L, H]
    input_ids: torch.Tensor,      # [B, L]
    attn_mask: torch.Tensor,      # [B, L]
    tokenizer: AutoTokenizer
) -> torch.Tensor:
    mask = attn_mask.clone().bool()  # [B, L]
    special_ids = set()
    for attr in ("cls_token_id", "bos_token_id", "eos_token_id", "sep_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)
    if special_ids:
        for tid in special_ids:
            mask = mask & (input_ids != tid)

    row_all_false = (~mask).all(dim=1)
    if row_all_false.any():
        mask[row_all_false] = attn_mask[row_all_false].bool()
        row_all_false2 = (~mask).all(dim=1)
        if row_all_false2.any():
            mask[row_all_false2] = True

    mask_f = mask.unsqueeze(-1).type_as(hidden)        # [B, L, 1]
    summed = (hidden * mask_f).sum(dim=1)              # [B, H]
    lens   = mask_f.sum(dim=1).clamp(min=1e-6)         # [B, 1]
    mean_pool = summed / lens                          # [B, H]

    very_neg = torch.finfo(hidden.dtype).min if hidden.dtype.is_floating_point else -1e9
    hidden_masked = hidden.masked_fill(~mask.unsqueeze(-1), very_neg)
    max_pool, _ = hidden_masked.max(dim=1)             # [B, H]

    return torch.cat([mean_pool, max_pool], dim=-1)    # [B, 2H]

@torch.no_grad()
def extract_features_concat_mean_max(
    sequences: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 16,
    use_fp16: bool = False
) -> torch.Tensor:
    feats = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        if use_fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)

        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        pooled = masked_mean_max(hidden, input_ids, attn_mask, tokenizer)  # [B, 2H]
        feats.append(pooled.detach().cpu())

    return torch.cat(feats, dim=0)

@torch.no_grad()
def esmfeature(
    sequences: List[str],
    esm_dir: Optional[str] = None,
    batch_size: int = 16,
    use_fp16: bool = False,
    device: Optional[torch.device] = None,
    return_tensor: bool = False
):

    tokenizer, model, device = get_tokenizer_and_model(esm_dir=esm_dir, device=device)
    feats_t = extract_features_concat_mean_max(
        sequences, tokenizer, model, device,
        batch_size=batch_size, use_fp16=use_fp16
    )  # torch.Tensor [N, 2H]
    return feats_t if return_tensor else feats_t.numpy()
