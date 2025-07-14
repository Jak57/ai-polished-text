from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import gc


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
            pass


def get_model_and_tokenizer(model_path):
    """
    Loads model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
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
