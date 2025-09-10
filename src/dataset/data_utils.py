from typing import Dict, List

from transformers import PreTrainedTokenizerFast

from src.utils.constant import IM_START_TOKEN

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_tokenize_qwen(conversations: List[Dict[str, str]], tokenizer: PreTrainedTokenizerFast) -> List[str]:
    roles = {"human": "user", "gpt": "assistant"}
    # char_template = (
    #         '{{- range $i, $_ := .Messages }}'
    #         '{{- $last := eq (len (slice $.Messages $i)) 1 }}'
    #         '{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user'
    #         '{{ .Content }}<end_of_turn>'
    #         '{{ if $last }}<start_of_turn>model'
    #         '{{ end }}'
    #         '{{- else if eq .Role "assistant" }}<start_of_turn>model'
    #         '{{ .Content }}{{ if not $last }}<end_of_turn>'
    #         '{{ end }}'
    #         '{{- end }}'
    #         '{{- end }}'
    #     )
    # tokenizer.chat_template = char_template

    default_system_message = [{
        "role": "system", 
        "content": "You are a helpful assistant."
    }]

    system_message = default_system_message.copy()
    try:
        if conversations[0]["role"] == "system":
            conversation = conversations.pop(0)
            role = conversation["role"]
            content = conversation["content"]
            system_message = [{"role": roles.get(role, role),  "content": content}]
    except:
        if conversations[0]["from"] == "system":
            conversation = conversations.pop(0)
            role = conversation["from"]
            content = conversation["value"]
            system_message = [{"role": roles.get(role, role),  "content": content}]

    system_prompt: str = tokenizer.apply_chat_template(system_message, tokenize=False, enable_thinking=False)
    package_prompt = []
    for idx, conversation in enumerate(conversations):
        try:
            role = conversation["role"]
            content = conversation["content"]
        except:
            role = conversation["from"]
            content = conversation["value"]
        
        if roles.get(role, role) == "user":
            human_prompt = tokenizer.apply_chat_template([{"role": roles.get(role, role), "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            # human_prompt = human_prompt[len(default_system_message):]
            package_prompt.append((system_prompt + human_prompt) if idx == 0 else human_prompt)
            # manual write prompt no use template

        elif roles.get(role, role) == "assistant":
            gpt_prompt = tokenizer.apply_chat_template([{"role": roles.get(role, role), "content": content}], tokenize=False, enable_thinking=False)
            prefix_gpt_prompt = IM_START_TOKEN + roles.get(role, role) + "\n"
            gpt_prompt = gpt_prompt[len(prefix_gpt_prompt):]
            package_prompt.append(gpt_prompt)

    return package_prompt

def main():
    pass

if __name__ == "__main__":
    main()