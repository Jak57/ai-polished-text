import os
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
os.environ["DISABLE_FLASH_ATTN"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import pandas as pd
import torch
import gc

# import os
# os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"


def get_length_of_text(text):
    """
    Counts the number of words in the text.
    """
    return len(text.split())


def get_message_template_for_degree_based_polishing(text, model_type="llama", polish_type="extremely minor"):
    """
    Gets the message template for the degree-bashed polishing.
    """
    type_to_name = {
        "extreme_minor" : "extremely minor",
        "minor"         : "minor",
        "slight_major"  : "slight major",
        "major"         : "major"
    }

    if model_type == "llama2":
        return [
            {
                "role": "system",
                "content": "You are a helpful chatbot who always responds with helpful information. You are asked to provide a polished version of an original text. Only generate the polished text.\n"
            },
            {
                "role": "user",
                "content": f"Polish the given original text below with {type_to_name[polish_type]} polishing. The difference between original and polished text must be {type_to_name[polish_type]}. The semantic meaning of polished text must be the same as original text. Just output the polished text, nothing else. The given original text:\n\"{text}\""
            }
        ]

def generate_response(model, tokenizer, message, max_new_tokens=100):
    """
    Generate a response to a given message using the model and tokenizer.
    message: a dict with 'role' and 'content' keys so that chat templates can be applied.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    
    try:
        # Attempt to use `apply_chat_template` (preferred for chat models)
        encodings = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,  # Ensure uniform input size
            truncation=True
        )

        if isinstance(encodings, dict):
            input_ids = encodings["input_ids"].to(model.device)
            attention_mask = encodings["attention_mask"].to(model.device)
        else:
            input_ids = encodings.to(model.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(model.device)  # Fallback mask

    except:
        input_ids = tokenizer.encode(message, return_tensors="pt").to(model.device)

    # Handle LLaMA and other models' stop tokens
    if 'llama' in model.config.model_type:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = tokenizer.eos_token_id
    
    if attention_mask is not None:
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            # pad_token_id=pad_token_id,
            pad_token_id=tokenizer.pad_token_id,

            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    else:
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    response = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response, skip_special_tokens=True)

    print("message:", message)
    print("response:", response_text)
    print("\n\n")
    return response_text


def generate_response2(model, tokenizer, message, max_new_tokens=100):
    """
    Generate a response to a given message using the model and tokenizer.
    message: a dict with 'role' and 'content' keys so that chat templates can be applied.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    try:
        # Attempt to use `apply_chat_template` (preferred for chat models)
        encodings = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        if isinstance(encodings, dict):
            input_ids = encodings["input_ids"].to(model.device)
            attention_mask = encodings["attention_mask"].to(model.device)
        else:
            input_ids = encodings.to(model.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(model.device) # Fallback mass
    except:
        input_ids = tokenizer.encode(message, return_tensors="pt").to(model.device)

    # Handle LLaMa and other model's stop tokens
    if 'llama' in model.config.model_type:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>") # End of turn 
        ]
        # print("------------> ", terminators)

    if attention_mask is not None:
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        print(f"attention mask is {attention_mask}")
        print(f"output is {outputs}")
    else:
        # TODO
        pass


def polish_text(data, editing_type, polish_type, polish_ratio, model_name, model_type, model, tokenizer):
    """
    Polishes the HWT dataset.
    """
    PERCENTAGE_BASED_POLISHING_TYPE      = 1  
    DEGREE_BASED_POLISHING_TYPE          = 2

    for i, row in data.iterrows():
        text = row['generation']
        if editing_type == PERCENTAGE_BASED_POLISHING_TYPE:
            pass
        elif editing_type == DEGREE_BASED_POLISHING_TYPE:
            message = get_message_template_for_degree_based_polishing(text, model_type, polish_type)

        if message == None:
            continue
        if model_type == "llama2":
            # pass
            response = generate_response(model, tokenizer, message, get_length_of_text(text)*1.5)


def get_model_and_tokenizer(model_path):
    """
    Loads model and tokenizer.
    """

#     bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,  # or load_in_4bit=True
#     llm_int8_threshold=6.0,
# )

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     quantization_config=bnb_config,
# )

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16
    )

            # torch_dtype
        # quantization_config=bnb_config


    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float32)
    return model, tokenizer


def main():
    PERCENTAGE_BASED_POLISHING_TYPE      = 1  
    DEGREE_BASED_POLISHING_TYPE          = 2
    perform_percentage_based_polishing   = False
    perform_degree_based_polishing       = True
    load_meta_llama_3_8B_instruct        = True
    # Load dataset
    data = pd.read_csv("HWT_original_data.csv")[:5]

    # Load model
    model_name, model_type, model, tokenizer = None, None, None, None
    if load_meta_llama_3_8B_instruct:
        model_name = "Meta-Llama-2-7b-chat"
        model_type = "llama2"
        model_path = "/home/ipt/jakir/projects/Adversarial_LLM/models/Llama-2-7b-chat-hf" 
        model, tokenizer = get_model_and_tokenizer(model_path)
        print(f"{model_name}: Successfully loaded model and tokenizer.")

    # Perform polishing
    if perform_degree_based_polishing:
        polish_type_list = ["extreme_minor", "minor", "slight_major", "major"]
        for polish_type in polish_type_list:
            print(f"Processing for polish type: {polish_type} ...")
            polish_text(data, DEGREE_BASED_POLISHING_TYPE, polish_type, None, model_name, model_type, model, tokenizer)


    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
