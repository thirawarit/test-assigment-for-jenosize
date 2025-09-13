import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

_model_cache = {}

def get_pipeline(model_id: str):
    """Load and cache Hugging Face pipeline

    Args:
        model_id (str): repo_id
    """

    if model_id not in _model_cache:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model_cache[model_id] = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    return _model_cache[model_id]

def get_generate(prompt: str, model_id: str, enable_thinking: bool = False):
    """_summary_

    Args:
        prompt (str): _description_
        model_id (str): _description_
        enable_thinking (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    if model_id not in _model_cache:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
        
    output = {}
    if enable_thinking:
        output["thinking_content"] = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    output["content"] = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return output




