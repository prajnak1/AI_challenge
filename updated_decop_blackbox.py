import pandas as pd
from oversample_labels_fn import generate_permutations
import sys
import os
from tqdm import tqdm
from torch import nn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import textwrap
from skopt import gp_minimize
from skopt.space import Integer

def get_prompt(instruction):
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")

def parse_text(text):
    wrapped_text = textwrap.fill(text, width=100)
    return wrapped_text

def generate(text, max_new_tokens, score_index):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.float16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        try:
            outputs = model.generate(**inputs,
                                     max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     eos_token_id=model.config.eos_token_id,
                                     pad_token_id=model.config.eos_token_id,
                                     return_dict_in_generate=True,
                                     output_scores=True,)

            final_outputs = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
            final_outputs = cut_off_text(final_outputs, '</s>')
            final_outputs = remove_substring(final_outputs, prompt)
            try:
                a = outputs["scores"][score_index][0][tokenizer("A").input_ids[-1]]
                b = outputs["scores"][score_index][0][tokenizer("B").input_ids[-1]]
                c = outputs["scores"][score_index][0][tokenizer("C").input_ids[-1]]
                d = outputs["scores"][score_index][0][tokenizer("D").input_ids[-1]]
            except Exception as e:
                print("Error in Probabilities")
                result = {"Text Output": "None", "A_Logit": 0, "B_Logit": 0, "C_Logit": 0, "D_Logit": 0}
                return result
        except Exception as e:
            print("CUDA out of memory error, skipping here", e)
            result = {"Text Output": "None", "A_Logit": 0, "B_Logit": 0, "C_Logit": 0, "D_Logit": 0}
            return result

    result = {"Text Output": final_outputs,
              "A_Logit": a,
              "B_Logit": b,
              "C_Logit": c,
              "D_Logit": d}
    return result

def extract_float_values(tensor_list):
    float_values = [tensor_item.item() for tensor_item in tensor_list]
    return float_values

def Query_LLM(data_type, query_data, document_name, author_name, max_new_tokens, score_index):
    if data_type == "BookTection":
        extra_prompt = f"""Question: Which of the following passages is verbatim from the \"{document_name}\" book by {author_name}?\nOptions:\n"""
    elif data_type == "arXivTection":
        extra_prompt = f"""Question: Which of the following passages is verbatim from the arXiv paper \"{document_name}\"?\nOptions:\n"""

    prompt = extra_prompt +  'A. ' + query_data[0] + '\n' + 'B. ' + query_data[1] + '\n' + 'C. ' + query_data[2] + '\n' + 'D. ' + query_data[3] + '\n\n' + 'Answer:'
    generated_text = generate(prompt, max_new_tokens, score_index)
    return generated_text

def process_files(data_type, passage_size, model, max_new_tokens, score_index):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if data_type == "BookTection":
        document = load_dataset("avduarte333/BookTection")
    else:
        document = load_dataset("avduarte333/arXivTection")

    document = pd.DataFrame(document["train"])
    unique_ids = document['ID'].unique().tolist()
    if data_type == "BookTection":
        document = document[document['Length'] == passage_size]
        document = document.reset_index(drop=True)

    model.eval()
    softmax = nn.Softmax(dim=0)
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for i in tqdm(range(len(unique_ids))):
        document_name = unique_ids[i]
        if data_type == "BookTection":
            out_dir = os.path.join(script_dir, f'DECOP_{data_type}_{passage_size}')
        else:
            out_dir = os.path.join(script_dir, f'DECOP_{data_type}')

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if data_type == "BookTection":
            fileOut = os.path.join(out_dir, f'{document_name}_Paraphrases_Oversampling_{passage_size}.xlsx')
        else:
            fileOut = os.path.join(out_dir, f'{document_name}_Paraphrases_Oversampling.xlsx')

        if os.path.exists(fileOut):
            document_aux = pd.read_excel(fileOut)
        else:
            document_aux = document[(document['ID'] == unique_ids[i])]
            document_aux = document_aux.reset_index(drop=True)
            document_aux = generate_permutations(document_df=document_aux)

        A_probabilities, B_probabilities, C_probabilities, D_probabilities, Max_Label = ([] for _ in range(5))

        if data_type == "BookTection":
            parts = document_name.split('_-_')
            document_name = parts[0].replace('_', ' ')
            author_name = parts[1].replace('_', ' ')
            print(f"Starting book - {document_name} by {author_name}")
        else:
            author_name = ""

        with torch.no_grad():
            for j in tqdm(range(len(document_aux))):
                result = Query_LLM(data_type=data_type, query_data=document_aux.iloc[j], document_name=document_name, author_name=author_name, max_new_tokens=max_new_tokens, score_index=score_index)
                final_output = result["Text Output"]
                final_output = final_output.strip()

                a = result["A_Logit"]
                b = result["B_Logit"]
                c = result["C_Logit"]
                d = result["D_Logit"]

                logits = torch.tensor([a, b, c, d], dtype=torch.float32)
                probabilities = softmax(logits)

                A_probabilities.append(probabilities[0])
                B_probabilities.append(probabilities[1])
                C_probabilities.append(probabilities[2])
                D_probabilities.append(probabilities[3])
                Max_Label.append(mapping.get(torch.argmax(probabilities).item(), 'Unknown'))
                torch.cuda.empty_cache()

            float_list1 = extract_float_values(A_probabilities)
            float_list2 = extract_float_values(B_probabilities)
            float_list3 = extract_float_values(C_probabilities)
            float_list4 = extract_float_values(D_probabilities)

        document_aux[f"A_Probability_{model_args_name}"] = float_list1
        document_aux[f"B_Probability_{model_args_name}"] = float_list2
        document_aux[f"C_Probability_{model_args_name}"] = float_list3
        document_aux[f"D_Probability_{model_args_name}"] = float_list4
        document_aux[f"Max_Label_NoDebias_{model_args_name}"] = Max_Label
        document_aux.to_excel(fileOut, index=False)
        print(f"Completed book - {document_name}!")

def evaluate_model(params):
    max_new_tokens, score_index = params
    # Define your own evaluation logic, e.g., return the average accuracy or loss
    # For simplicity, let's assume a dummy evaluation returning random performance
    # Replace this with your actual evaluation
    return -1 * dummy_evaluation_performance

def optimize_hyperparameters():
    space = [
        Integer(1, 10, name='max_new_tokens'),
        Integer(0, 3, name='score_index')
    ]
    res = gp_minimize(evaluate_model, space, n_calls=20, random_state=0)
    return res

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python <name_of_file.py> --data <data_file> --target_model <model_name> [--length <passage_size>]")
        print("<passage_size> is only mandatory for BookTection and should be one of: <small>, <medium>, or <large>")
        sys.exit(1)

    data_type = sys.argv[2]
    target_model = sys.argv[4]
    passage_size = None
    if data_type == "BookTection":
        if len(sys.argv) < 7:
            print("Usage: python <name_of_file.py> --data <data_file> --target_model <model_name> [--length <passage_size>]")
            print("<passage_size> is only mandatory for BookTection and should be one of: <small>, <medium>, or <large>")
            sys.exit(1)
        passage_size = sys.argv[6]

    model_args_name = target_model
    print(f"Loading {model_args_name}")

    tokenizer = AutoTokenizer.from_pretrained(target_model)
    model = AutoModelForCausalLM.from_pretrained(target_model, device_map="auto")

    optimized_params = optimize_hyperparameters()
    print(f"Optimized Parameters: max_new_tokens={optimized_params[0]}, score_index={optimized_params[1]}")

    process_files(data_type, passage_size, model, optimized_params[0], optimized_params[1])
