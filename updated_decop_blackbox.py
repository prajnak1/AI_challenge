import pandas as pd
import sys
import os
from tqdm import tqdm
from torch import nn
import torch
from openai import OpenAI
# from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from datasets import load_dataset
from oversample_labels_fn import generate_permutations

softmax = nn.Softmax(dim=0)
mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

QA_PROMPT = "You are entering a multiple choice questions exam. You should directly answer each question by choosing the correct option. Be concise and straight to the point in your answer. Output only the letter corresponding to the correct answer."

class EnhancedNN(nn.Module):
    def __init__(self):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def query_llm(data_type, model_name, query_data, document_name, author_name, client=None):
    if data_type == "BookTection":
        extra_prompt = f"Question: Which of the following passages is verbatim from the "{document_name}" book by {author_name}?\nOptions:\n""
    elif data_type == "arXivTection":
        extra_prompt = f"Question: Which of the following passages is verbatim from the arXiv paper {document_name}?""

    if model_name == "ChatGPT":
        prompt = extra_prompt + 'A. ' + query_data[0] + '\n' + 'B. ' + query_data[1] + '\n' + 'C. ' + query_data[2] + '\n' + 'D. ' + query_data[3] + '\n' + 'Answer: '
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            seed=2319,
            logprobs=4,
            logit_bias={32: +100, 33: +100, 34: +100, 35: +100}  # Increase probabilities of tokens A, B, C, D equally
        )
        dict_probs = response.choices[0].logprobs.top_logprobs[0]
        logits = torch.tensor([dict_probs.get("A", 0.0), dict_probs.get("B", 0.0), dict_probs.get("C", 0.0), dict_probs.get("D", 0.0)], dtype=torch.float32)
        probabilities = softmax(logits)
        return probabilities
    else:
        prompt = QA_PROMPT + extra_prompt + 'A. ' + query_data[0] + '\n' + 'B. ' + query_data[1] + '\n' + 'C. ' + query_data[2] + '\n' + 'D. ' + query_data[3]
        completion = client.completions.create(
            model="claude-2",
            max_tokens_to_sample=1,
            prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT} Answer: ",
            temperature=0)
        return completion.completion.strip()

def extract_float_values(tensor_list):
    return [tensor_item.item() for tensor_item in tensor_list]

def process_files(data_type, passage_size, model, client=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = "avduarte333/BookTection" if data_type == "BookTection" else "avduarte333/arXivTection"
    document = load_dataset(dataset_name)["train"]
    document = pd.DataFrame(document)
    unique_ids = document['ID'].unique().tolist()

    if data_type == "BookTection":
        document = document[document['Length'] == passage_size].reset_index(drop=True)

    for i in tqdm(range(len(unique_ids))):
        document_name = unique_ids[i]
        out_dir = os.path.join(script_dir, f'DECOP_{data_type}_{passage_size if data_type == "BookTection" else ""}')
        os.makedirs(out_dir, exist_ok=True)
        file_out = os.path.join(out_dir, f'{document_name}_Paraphrases_Oversampling_{passage_size if data_type == "BookTection" else ""}.xlsx')

        if os.path.exists(file_out):
            document_aux = pd.read_excel(file_out)
        else:
            document_aux = document[document['ID'] == unique_ids[i]].reset_index(drop=True)
            document_aux = generate_permutations(document_df=document_aux)

        A_probabilities, B_probabilities, C_probabilities, D_probabilities, Max_Label = ([] for _ in range(5))

        if data_type == "BookTection":
            parts = document_name.split('_-_')
            document_name = parts[0].replace('_', ' ')
            author_name = parts[1].replace('_', ' ')
            print(f"Starting book - {document_name} by {author_name}")
        else:
            author_name = ""

        if model == "ChatGPT":
            for j in tqdm(range(len(document_aux))):
                probabilities = query_llm(data_type, model, document_aux.iloc[j], document_name, author_name, client)
                A_probabilities.append(probabilities[0])
                B_probabilities.append(probabilities[1])
                C_probabilities.append(probabilities[2])
                D_probabilities.append(probabilities[3])
                Max_Label.append(mapping.get(torch.argmax(probabilities).item(), 'Unknown'))

            document_aux["A_Probability"] = extract_float_values(A_probabilities)
            document_aux["B_Probability"] = extract_float_values(B_probabilities)
            document_aux["C_Probability"] = extract_float_values(C_probabilities)
            document_aux["D_Probability"] = extract_float_values(D_probabilities)
            document_aux["Max_Label_NoDebias"] = Max_Label
        else:
            for j in tqdm(range(len(document_aux))):
                Max_Label.append(query_llm(data_type, model, document_aux.iloc[j], document_name, author_name, client))
            document_aux["Claude2.1"] = Max_Label

        document_aux.to_excel(file_out, index=False)
        print(f"Completed book - {document_name}!")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python <name_of_file.py> --data <data_file> --target_model <model_name> [--length <passage_size>]")
        print("<passage_size> is only mandatory for BookTection and should be one of: <small>, <medium>, or <large>")
        sys.exit(1)

    data_index = sys.argv.index("--data")
    model_index = sys.argv.index("--target_model")

    data_type = sys.argv[data_index + 1]
    model = sys.argv[model_index + 1]

    if model == "ChatGPT":
        api_key = os.getenv("OPENAI_API_KEY")  # Use environment variable for API key
        if not api_key:
            raise ValueError("API key for OpenAI is not set.")
        client = OpenAI(api_key=api_key)
    # elif model == "Claude":
        # claude_api_key = os.getenv("CLAUDE_API_KEY")  # Use environment variable for API key
        # if not claude_api_key:
        #     raise ValueError("API key for Claude is not set.")
        # client = Anthropic(api_key=claude_api_key)
    else:
        print("Available models are: <ChatGPT> or <Claude>")
        sys.exit()

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
        passage_size = "default_value"  # Replace with an appropriate default value
    else:
        print("Invalid data_file. Available options are: BookTection or arXivTection")
        sys.exit(1)

    process_files(data_type, passage_size, model, client)
