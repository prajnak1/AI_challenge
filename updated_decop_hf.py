import pandas as pd
import sys
import os
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import textwrap

# Constants for system prompts
B_SYS = "<<SYS>>\n"
E_SYS = "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are entering a multiple choice questions exam. You should directly answer each question by choosing the correct option. Be concise and straight to the point in your answer. Output only the letter corresponding to the correct answer.

Format your answer as '<correct letter>'."""

# Initialize tokenizer and model
access_token = "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"
tokenizer = None
model = None

# Function to initialize tokenizer and model
def initialize_model(model_args_name):
    global tokenizer, model
    model_names = {
        "LLaMA2-70B": "meta-llama/Llama-2-70b-chat-hf",
        "LLaMA2-13B": "meta-llama/Llama-2-13b-chat-hf",
        "LLaMA2-7B": "meta-llama/Llama-2-7b-chat-hf",
    }
    if model_args_name not in model_names:
        print("Available models are: LLaMA2-70B, LLaMA2-13B, or LLaMA2-7B")
        sys.exit(1)
    
    model_name = model_names[model_args_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

# Function to generate text using the model
def generate(prompt, max_new_tokens=2, score_index=1):
    global tokenizer, model
    
    # Wrap prompt and generate input tensors
    wrapped_prompt = B_SYS + DEFAULT_SYSTEM_PROMPT + prompt + E_SYS
    inputs = tokenizer(wrapped_prompt, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to('cuda' if torch.cuda.is_available() else 'cpu') for key, val in inputs.items()}
    
    try:
        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(**inputs,
                                     max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     eos_token_id=model.config.eos_token_id,
                                     pad_token_id=model.config.eos_token_id,
                                     return_dict_in_generate=True,
                                     output_scores=True)
            
            # Decode outputs and remove unnecessary tokens
            decoded_output = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            decoded_output = decoded_output.replace(prompt, "").strip()
            
            # Extract logits and compute probabilities
            logits = outputs.scores[score_index][0]
            probabilities = nn.functional.softmax(logits, dim=0)
            a_prob, b_prob, c_prob, d_prob = probabilities.tolist()
            
            return {
                "Text Output": decoded_output,
                "A_Logit": logits[tokenizer("A").input_ids[-1]].item(),
                "B_Logit": logits[tokenizer("B").input_ids[-1]].item(),
                "C_Logit": logits[tokenizer("C").input_ids[-1]].item(),
                "D_Logit": logits[tokenizer("D").input_ids[-1]].item(),
                "A_Probability": a_prob,
                "B_Probability": b_prob,
                "C_Probability": c_prob,
                "D_Probability": d_prob
            }
    
    except Exception as e:
        print(f"Error in generation: {e}")
        return {
            "Text Output": "None",
            "A_Logit": 0,
            "B_Logit": 0,
            "C_Logit": 0,
            "D_Logit": 0,
            "A_Probability": 0,
            "B_Probability": 0,
            "C_Probability": 0,
            "D_Probability": 0
        }

# Function to process files and evaluate passages
def process_files(data_type, passage_size, model_args_name):
    global tokenizer, model
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = "avduarte333/BookTection" if data_type == "BookTection" else "avduarte333/arXivTection"
    document = load_dataset(dataset_name)["train"]
    document = pd.DataFrame(document)
    unique_ids = document['ID'].unique().tolist()
    
    if data_type == "BookTection":
        document = document[document['Length'] == passage_size].reset_index(drop=True)
    
    softmax = nn.Softmax(dim=0)
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    for unique_id in tqdm(unique_ids):
        if data_type == "BookTection":
            out_dir = os.path.join(script_dir, f'DECOP_{data_type}_{passage_size}')
        else:
            out_dir = os.path.join(script_dir, f'DECOP_{data_type}')
        
        os.makedirs(out_dir, exist_ok=True)
        file_out = os.path.join(out_dir, f'{unique_id}_Paraphrases_Oversampling_{passage_size}.xlsx' if data_type == "BookTection" else f'{unique_id}_Paraphrases_Oversampling.xlsx')
        
        if os.path.exists(file_out):
            document_aux = pd.read_excel(file_out)
        else:
            document_aux = document[document['ID'] == unique_id].reset_index(drop=True)
            document_aux = generate_permutations(document_df=document_aux)
        
        a_probabilities, b_probabilities, c_probabilities, d_probabilities, max_labels = [], [], [], [], []
        
        with torch.no_grad():
            for index, row in tqdm(document_aux.iterrows(), total=len(document_aux)):
                query_data = [row['A'], row['B'], row['C'], row['D']]
                document_name = row['ID']
                author_name = row.get('Author', '')  # Assuming Author column exists

                result = Query_LLM(data_type, query_data, document_name, author_name)
                final_output = result["Text Output"].strip()

                a_logit = result["A_Logit"]
                b_logit = result["B_Logit"]
                c_logit = result["C_Logit"]
                d_logit = result["D_Logit"]

                logits = torch.tensor([a_logit, b_logit, c_logit, d_logit], dtype=torch.float32)
                probabilities = softmax(logits)

                a_probabilities.append(probabilities[0].item())
                b_probabilities.append(probabilities[1].item())
                c_probabilities.append(probabilities[2].item())
                d_probabilities.append(probabilities[3].item())
                max_labels.append(mapping.get(torch.argmax(probabilities).item(), 'Unknown'))

        document_aux[f"A_Probability_{model_args_name}"] = a_probabilities
        document_aux[f"B_Probability_{model_args_name}"] = b_probabilities
        document_aux[f"C_Probability_{model_args_name}"] = c_probabilities
        document_aux[f"D_Probability_{model_args_name}"] = d_probabilities
        document_aux[f"Max_Label_NoDebias_{model_args_name}"] = max_labels
        
        document_aux.to_excel(file_out, index=False)
        print(f"Completed processing {unique_id}.")

# Function to query the Language Model
def Query_LLM(data_type, query_data, document_name, author_name):
    if data_type == "BookTection":
        extra_prompt = f"Question: Which of the following passages is verbatim from the \"{document_name}\" book by {author_name}?\nOptions:\n"
    elif data_type == "arXivTection":
        extra_prompt = f"Question: Which of the following passages is verbatim from the arXiv paper \"{document_name}\"?\nOptions:\n"
    
    prompt = extra_prompt + 'A. ' + query_data[0] + '\n' + 'B. ' + query_data[1] + '\n' + 'C. ' + query_data[2] + '\n' + 'D. ' + query_data[3] + '\n\nAnswer:'

    return generate(prompt)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python <name_of_file.py> --data <data_file> --target_model <model_name> [--length <passage_size>]")
        print("<passage_size> is only mandatory for BookTection and should be one of: <small>, <medium>, or <large>")
        sys.exit(1)

    data_index = sys.argv.index("--data")
    model_index = sys.argv.index("--target_model")
    
    data_type = sys.argv[data_index + 1]
    model_args_name = sys.argv[model_index + 1]

    initialize_model(model_args_name)

    if data_type == "BookTection":
        if "--length" not in sys.argv:
            print("Passage size (--length) is mandatory for BookTection data.")
            sys.exit(1)
        passage_size_index = sys.argv.index("--length")
        passage_size = sys.argv[passage_size_index + 1]

        if passage_size not in ["small", "medium", "large"]:
            print("Invalid passage_size. Available options are: <small>, <medium>, or <large>")
            sys.exit(1)
    elif data_type == "arXivTection":
        passage_size = "default_value"  # Set a default value for arXivTection data
    else:
        print("Invalid data_file. Available options are: BookTection or arXivTection")
        sys.exit(1)

    process_files(data_type, passage_size, model_args_name)
