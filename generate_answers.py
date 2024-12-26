import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Features, Value
import argparse
from tqdm import tqdm


features = Features({
    "question_id": Value("string"),
    "Question": Value("string"),
    "A": Value("string"),
    "B": Value("string"),
    "C": Value("string"),
    "D": Value("string")
})

def collate_fn(batch):
    questions = [item['Question'] for item in batch]
    options = [{
        'A': item['A'],
        'B': item['B'],
        'C': item['C'],
        'D': item['D']
    } for item in batch]
    ids = [item['question_id'] for item in batch]
    
    return {
        'questions': questions,
        'options': options,
        'ids': ids
    }

def format_prompt(question, options):
    prompt = "Answer the following multiple choice question and provide only letter of the correct answer.\n"
    prompt += f"{question.strip()}\n"
    for key in ['A', 'B', 'C', 'D']:
        if key in options:
            prompt += f"{key}. {options[key]}\n"
    prompt += "Answer：\n"
    return prompt

def evaluate(model, tokenizer, dataloader, device):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):

            prompts = [format_prompt(q, opt) for q, opt in zip(batch['questions'], batch['options'])]
            

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            


            outputs = model.generate(**inputs, max_new_tokens=5)
            

            new_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            

            responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            
            for question_id, response in zip(batch['ids'], responses):
                answer = response.strip()
                results.append({
                    'question_id': question_id,
                    'predicted_answer': answer
                })
    
    return results

def main(args):
    dataset = load_dataset("MBZUAI-LLM/Mobile-MMLU", "all", split="test", features=features)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # hf requires that pad_token is set or generate will crash 
    # set to safe default if model tokenizer does not set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # normalize args
    if isinstance(args.device, str):
        args.device = args.device.lower()

    if isinstance(args.backend, str):
        args.backend = args.backend.lower()

    if args.backend == 'gptqmodel':
        try:
            from gptqmodel import GPTQModel
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "Tried to load gptqmodel, but gptqmodel is not installed ",
                "please install gptqmodel via `pip install gptqmodel --no-build-isolation`",
            )
        model = GPTQModel.load(model_id_or_path=args.model_name, device=args.device if args.device != "auto" else : None)
        device = model.device
    else:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        ).eval()

    print(f"Using device: {device}")
    print(f"Processing {len(dataset)} examples in batches of {args.batch_size}...")
    results = evaluate(model, tokenizer, dataloader, device)
    
    output_file = f"{args.model_name.replace('/', '_')}_predictions.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name or path of the model to use')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing (default: 32)')
    parser.add_argument('--device', type=str, default="auto",
                      help='Device to run the model on (default: `auto` = use cuda if available else cpu)')
    parser.add_argument('--backend', type=str, default="hf",
                        help='Load Model on (default: `hf`). Use `gptqmodel` for gptq quantized models.')
    args = parser.parse_args()
    
    main(args)
