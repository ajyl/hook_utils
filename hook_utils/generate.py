import torch
from hook_utils.record_utils import record_activations


@torch.no_grad()
def generate_simple(model, tokenizer, prompts, max_prompt_length=32, max_new_tokens=16):
    """
    Simplified version of model.generate, without all the
    HuggingFace complexity.
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(model.device)

    timestep = 0

    input_ids = inputs.input_ids
    while timestep < max_new_tokens:
        logits = model(**inputs).logits
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)

        if ((input_ids == tokenizer.eos_token_id).sum(dim=1) == 1).all():
            break

        timestep += 1

    return input_ids


@torch.no_grad()
def hooked_generate_topk(
    model,
    tokenizer,
    prompts,
    record_module_names,
    k=10,
    max_prompt_length=32,
    max_new_tokens=16,
):
    """
    Simplified version of model.generate, without all the
    HuggingFace complexity.
    """
    eos_token_id = tokenizer.eos_token_id
    device = model.device
    batch_size = len(prompts)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(device)

    timestep = 0

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    acts = {module_name: [] for module_name in record_module_names}
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    valid_timestep_mask = []
    while timestep < max_new_tokens:
        if finished.all():
            break

        alive = ~finished
        valid_timestep_mask.append(alive.clone())

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        with record_activations(model, record_module_names) as cache:
            logits = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits

        for module_name in record_module_names:
            _acts = cache[module_name][0][:, -1].clone()
            _acts[~alive] = torch.nan
            acts[module_name].append(_acts)

        topk_logits, topk_indices = torch.topk(logits[:, -1], k=k, dim=-1)
        topk_probs = torch.softmax(topk_logits, dim=-1)
        next_token = torch.multinomial(topk_probs, num_samples=1)
        next_token = torch.gather(topk_indices, 1, next_token)

        reached_eos = next_token.squeeze(1) == eos_token_id
        new_finished = (~finished) & (reached_eos)
        finished |= new_finished
        next_token[finished] = eos_token_id

        input_ids = torch.cat((input_ids, next_token), dim=1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((batch_size, 1), device=attention_mask.device),
            ),
            dim=1,
        )

        timestep += 1

    acts = {
        module_name: torch.stack(acts[module_name], dim=1)
        for module_name in record_module_names
    }
    valid_timestep_mask = torch.stack(valid_timestep_mask, dim=1)
    return input_ids, acts, valid_timestep_mask
