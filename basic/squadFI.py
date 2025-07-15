import os
import random
import numpy as np
import torch
import time
import struct
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import argparse
from torch.utils.data import DataLoader
import evaluate
import re

# Set environment variables and seed for reproducibility
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.multiprocessing.set_start_method('spawn', force=True)

def seed_torch(seed=196):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()
device = "cuda"
torch.set_printoptions(threshold=np.inf) 

def perform_bit_flip_weight(tensor, bit_position):
    """Flip two bits of a weight tensor element."""
    with torch.no_grad():
        tensor_bf16 = tensor.to(torch.bfloat16)
        bits = tensor_bf16.view(torch.int16)
        mask = (1 << bit_position[0]) | (1 << bit_position[1])
        bits = bits ^ mask
        flipped_tensor = bits.view(torch.bfloat16)
    return flipped_tensor

def perform_bit_flip_neuron(tensor, bit_position):
    """Flip two bits of a neuron output element."""
    with torch.no_grad():
        tensor_bf16 = tensor.to(torch.bfloat16)
        bits = tensor_bf16.view(torch.int16)
        mask = (1 << bit_position[0]) | (1 << bit_position[1])
        bits = bits ^ mask
        flipped_tensor = bits.view(torch.bfloat16)
    return flipped_tensor

def perform_bit_flip_single(tensor, bit_position):
    """Flip a single bit of a neuron output element."""
    with torch.no_grad():
        tensor_bf16 = tensor.to(torch.bfloat16)
        bits = tensor_bf16.view(torch.int16)
        mask = (1 << bit_position)
        bits = bits ^ mask
        flipped_tensor = bits.view(torch.bfloat16)
    return flipped_tensor

def create_output_hook(module, bit_position, coordinates, token_position, fault_mode):
    """Create a hook function that will inject bit flips in the output tensor."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]  # Handle multiple output cases
        
        # Get current hook call count for this input
        if not hasattr(hook, 'count'):
            hook.count = 0
        
        # Only inject error at the token_position-th hook call
        if hook.count == token_position:
            x, y = coordinates
            if x < output.shape[1] and y < output.shape[2]:
                if fault_mode == 'neuron':
                    output[0, x, y] = perform_bit_flip_neuron(output[0, x, y], bit_position)
                elif fault_mode == 'single':
                    output[0, x, y] = perform_bit_flip_single(output[0, x, y], bit_position)
            else:
                print(f"Invalid coordinates: x={x}, y={y}, output shape={output.shape}")
        hook.count += 1
        return output
    return hook 

def record_output_dimensions(model, dataset, tokenizer, device):
    """Record dimension information during baseline evaluation"""
    sequence_lengths = []
    min_tokens = float('inf')
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if output.shape[1] > 1:
            sequence_lengths.append(output.shape[1])  # Record sequence_length dimension
    
    # Only register hook for the first module of the first layer
    first_layer = model.model.layers[0]
    module = first_layer.self_attn.v_proj  # Use the first module
    hook = module.register_forward_hook(hook_fn)
    
    # Run baseline evaluation
    for idx in range(len(dataset)):
        # Get input
        _, prompt, input_ids, _, _ = get_input(dataset, tokenizer, idx, device)
        prompt_len = len(input_ids[0])
        
        # Generate output
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=prompt_len + 50,
                do_sample=False,
                num_beams=1,  # greedy search
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Record number of generated tokens
        generated_tokens = len(output[0][prompt_len:])
        min_tokens = min(min_tokens, generated_tokens)
    
    # Remove hook
    hook.remove()
    
    # Calculate minimum sequence_length
    min_seq_len = min(sequence_lengths) if sequence_lengths else 1
    print(f"Minimum sequence length observed: {min_seq_len}")
    print(f"Minimum generated tokens: {min_tokens}")
    
    return min_seq_len, min_tokens

def get_input(dataset, tokenizer, id, device):
    """Prepare input for SQuAD dataset"""
    context = dataset[id]["context"]
    question = dataset[id]["question"]
    answer = dataset[id]["answers"]["text"][0] if dataset[id]["answers"]["text"] else ""
    
    # Add id and answer_start for later use
    answer_start = dataset[id]["answers"]["answer_start"][0] if dataset[id]["answers"]["answer_start"] else 0
    question_id = dataset[id]["id"]
    
    prompt = f"Based on the following context, answer the question:\nContext: {context}\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device, non_blocking=True)
    return answer, prompt, input_ids, question_id, answer_start

def generate(id, dataset, tokenizer, model, max_length=50):
    """Generate answer for SQuAD example using greedy search"""
    reference, prompt, input_ids, question_id, answer_start = get_input(dataset, tokenizer, id, device)
    prompt_len = len(input_ids[0])
    context = dataset[id]["context"]
    question = dataset[id]["question"]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=prompt_len + max_length,
            do_sample=False,
            num_beams=1,  # greedy search
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    
    # Process generated answer:
    # 1. If there is a newline, only keep the content before the newline
    if "\n" in generated_text:
        generated_text = generated_text.split("\n")[0].strip()
    
    # 2. If there is a period or other punctuation, only keep the first sentence
    first_sentence = ""
    sentence_endings = re.split(r'(?<=[.!?])\s+', generated_text)
    if sentence_endings and len(sentence_endings) > 0:
        first_sentence = sentence_endings[0].strip()
    else:
        first_sentence = generated_text.strip()
    
    # Clean up memory
    del output
    del input_ids
    torch.cuda.empty_cache()
    
    return first_sentence, reference, context, question_id, answer_start

def compute_metric_scores(prediction, reference, question_id, answer_start):
    """Compute metric scores for answer, using original dataset id and answer_start"""
    if not prediction or not reference:
        return {"exact_match": 0.0, "f1": 0.0}
    
    squad_metric = evaluate.load("squad")
    
    # Prepare input in SQuAD evaluation format, using original question ID
    prediction_dict = {
        "id": question_id,
        "prediction_text": prediction
    }
    
    reference_dict = {
        "id": question_id,
        "answers": {
            "text": [reference],
            "answer_start": [answer_start]  # Use original answer_start
        }
    }
    
    results = squad_metric.compute(
        predictions=[prediction_dict], 
        references=[reference_dict]
    )
    
    del squad_metric
    torch.cuda.empty_cache()
    return results

def read_ids_from_file(filename):
    """Read sample IDs from file"""
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file] 

def main():
    parser = argparse.ArgumentParser(description="SQuAD Fault Injection Experiment")
    parser.add_argument('--fault_mode', type=str, default='weight', 
                       choices=['weight', 'neuron', 'single'], 
                       help='Fault injection mode: weight (weight), neuron (neuron output double bit), single (neuron output single bit)')
    parser.add_argument('--model', type=str, default='qwen',
                       choices=['qwen', 'llama3', 'falcon'],
                       help='Model type: Qwen-7B, LLaMA3-8B, or Falcon3-7B')
    parser.add_argument('--num_trials', type=int, default=500,
                       help='Number of bit flip trials per sample')
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Create output directory (if it doesn't exist)
    output_dir = f"squad_{args.model}_{args.fault_mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file
    baseline_file = os.path.join(output_dir, "baseline_answers.txt")
    different_answers_file = os.path.join(output_dir, "different_answers.txt")
    
    # Open file for writing
    baseline_output = open(baseline_file, "w", encoding="utf-8")
    different_answers = open(different_answers_file, "w", encoding="utf-8")

    # Randomly select 100 input samples
    num_samples = 100

    # Load SQuAD v2 dataset
    print("Loading SQuAD v2 dataset...")
    dataset_full = datasets.load_dataset("squad_v2", split="validation")
    print(f"SQuAD v2 dataset loaded with {len(dataset_full)} samples")
    
    # Only keep samples with answers
    dataset_with_answers = dataset_full.filter(lambda example: len(example["answers"]["text"]) > 0)
    print(f"Filtered to {len(dataset_with_answers)} samples with answers")
    
    # Create random sample IDs within dataset size range
    max_id = len(dataset_with_answers)
    if max_id < num_samples:
        print(f"Warning: Only {max_id} samples available with answers. Using all of them.")
        num_samples = max_id
        
    sample_ids = random.sample(range(max_id), num_samples)

    # Write sample IDs to file for reproducibility
    with open(os.path.join(output_dir, "sample_indices.txt"), "w") as f:
        for i in sample_ids:
            f.write(f"{i}\n")

    # Select samples
    dataset = dataset_with_answers.select(sample_ids)
    print(f"Selected {len(dataset)} samples for experiments")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    global tokenizer

    # Load model and tokenizer based on selected model type
    if args.model == 'qwen':
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        model.resize_token_embeddings(len(tokenizer))
        num_layers = 28
        layer_weights = {
            'self_attn.v_proj': 1,
            'self_attn.k_proj': 1,
            'self_attn.q_proj': 7,
            'self_attn.o_proj': 7,
            'mlp.up_proj': 37,
            'mlp.gate_proj': 37,
            'mlp.down_proj': 37
        }
    elif args.model == 'llama3':
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        model.resize_token_embeddings(len(tokenizer))
        num_layers = 28
        layer_weights = {
            'self_attn.v_proj': 1,
            'self_attn.k_proj': 1,
            'self_attn.q_proj': 4,
            'self_attn.o_proj': 4,
            'mlp.up_proj': 14,
            'mlp.gate_proj': 14,
            'mlp.down_proj': 14
        }
    else:  # falcon
        model_name = "tiiuae/Falcon3-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        model.resize_token_embeddings(len(tokenizer))
        num_layers = 28
        layer_weights = {
            'self_attn.v_proj': 2,
            'self_attn.k_proj': 2,
            'self_attn.q_proj': 6,
            'self_attn.o_proj': 6,
            'mlp.up_proj': 45,
            'mlp.gate_proj': 45,
            'mlp.down_proj': 45
        }

    print(f"Model {args.model} loaded")
    print(f"Number of layers: {num_layers}")
    for layer in model.children():
        print(layer)

    # Compute total weight for weighted random selection
    total_weight = sum(layer_weights.values())
    layers = list(layer_weights.keys())
    weights = [layer_weights[layer] / total_weight for layer in layers]

    # Create results dictionary
    results = {
        "baseline_answers": [],
        "bit_flip_trials": [],
    }

    # Store baseline answers for later comparison
    baseline_answers = {}

    # First execute baseline answer generation (without bit flips)
    print("Generating baseline answers...")
    baseline_progress = tqdm(range(len(dataset)), desc="Baseline answers")

    total_baseline_exact_match = 0.0
    total_baseline_f1 = 0.0
    
    for idx in range(len(dataset)):
        answer, reference, context, question_id, answer_start = generate(idx, dataset, tokenizer, model)
        metric_scores = compute_metric_scores(answer, reference, question_id, answer_start)
        
        total_baseline_exact_match += metric_scores["exact_match"]
        total_baseline_f1 += metric_scores["f1"]

        results["baseline_answers"].append({
            "sample_id": idx,
            "question": dataset[idx]["question"],
            "reference": reference,
            "answer": answer,
            "exact_match": metric_scores["exact_match"],
            "f1": metric_scores["f1"]
        })
        
        # Write baseline answers to baseline file
        baseline_output.write(f"Sample ID: {idx} (Baseline)\n")
        baseline_output.write(f"Question: {dataset[idx]['question']}\n")
        baseline_output.write(f"Reference: {reference}\n")
        baseline_output.write(f"Answer: {answer}\n")
        baseline_output.write(f"Exact Match: {metric_scores['exact_match']:.4f}\n")
        baseline_output.write(f"F1: {metric_scores['f1']:.4f}\n")
        baseline_output.write("="*80 + "\n\n")
        
        # Store baseline answers for later comparison
        baseline_answers[idx] = answer

        baseline_progress.update(1)

    avg_baseline_exact_match = total_baseline_exact_match / len(dataset)
    avg_baseline_f1 = total_baseline_f1 / len(dataset)
    print(f"Baseline average Exact Match score: {avg_baseline_exact_match:.4f}")
    print(f"Baseline average F1 score: {avg_baseline_f1:.4f}")

    # Close baseline answers file
    baseline_output.close()
    
    if args.fault_mode in ["neuron", "single"]:
        min_seq_len, min_tokens = record_output_dimensions(model, dataset, tokenizer, device)

    # Perform bit flip experiments
    print(f"Performing {args.num_trials} bit flip trials...")
    bit_flip_progress = tqdm(total=len(dataset) * args.num_trials, desc="Bit flip trials")

    for trial in range(args.num_trials):
        # Select random layer index
        layer_idx = random.randint(0, num_layers - 1)  # Use loaded num_layers

        # Randomly select module based on weights
        selected_module = random.choices(layers, weights=weights)[0]

        # Navigate to target layer and module
        target_layer = model.model.layers[layer_idx]

        module_path = selected_module.split('.')
        current_module = target_layer
        for path_part in module_path:
            current_module = getattr(current_module, path_part)
            
        # Get weight tensor
        weight_tensor = current_module.weight
        
        # Set parameters based on fault mode
        if args.fault_mode == 'weight':
            x = random.randint(0, weight_tensor.shape[0] - 1)
            y = random.randint(0, weight_tensor.shape[1] - 1)
            bit_position = random.sample(range(16), 2)
            original_weight_value = weight_tensor[x, y].clone()
            with torch.no_grad():
                weight_tensor[x, y] = perform_bit_flip_weight(weight_tensor[x, y], bit_position)
        else:  # neuron or single
            if args.fault_mode == 'neuron':
                bit_position = random.sample(range(16), 2)
            else:  # single
                bit_position = random.randint(0, 15)
            token_position = random.randint(0, min_tokens - 1)
            if token_position == 0:
                x = random.randint(0, min_seq_len - 1)
            else:
                x = 0
            y = random.randint(0, weight_tensor.shape[0] - 1)
            hook = create_output_hook(current_module, bit_position, (x, y), token_position, args.fault_mode)
            hook_handle = current_module.register_forward_hook(hook)

        # Create current trial output file
        trial_file = os.path.join(output_dir, f"bit_flip_trial_{trial}_layer_{layer_idx}_module_{selected_module.replace('.', '_')}_bit_{bit_position}.txt")
        trial_output = open(trial_file, "w", encoding="utf-8")
        
        # Write current trial information
        trial_output.write(f"Trial: {trial}\n")
        trial_output.write(f"Layer: {layer_idx}\n")
        trial_output.write(f"Module: {selected_module}\n")
        trial_output.write(f"Bit Position: {bit_position}\n\n")

        # Generate answer using fault injection
        for sample_idx in range(len(dataset)):
            if args.fault_mode in ["neuron", "single"] and hasattr(hook, 'count'):
                delattr(hook, 'count')
                
            start_time = time.time()
            answer, reference, context, question_id, answer_start = generate(sample_idx, dataset, tokenizer, model)
            successful = True
            end_time = time.time()

            # Compute scores
            metric_scores = compute_metric_scores(answer, reference, question_id, answer_start)

            # Record results
            trial_result = {
                "sample_id": sample_idx,
                "question": dataset[sample_idx]["question"],
                "layer_idx": layer_idx,
                "module": selected_module,
                "bit_position": bit_position,
                "reference": reference,
                "answer": answer,
                "exact_match": metric_scores["exact_match"],
                "f1": metric_scores["f1"],
                "time_taken": end_time - start_time,
                "successful": successful
            }
            
            if args.fault_mode == 'weight':
                trial_result["original_weight_value"] = original_weight_value.item()
                trial_result["flipped_weight_value"] = weight_tensor[x, y].item()
            
            results["bit_flip_trials"].append(trial_result)
            
            # Write to current trial output file
            trial_output.write(f"Sample ID: {sample_idx}\n")
            trial_output.write(f"Question: {dataset[sample_idx]['question']}\n")
            trial_output.write(f"Reference: {reference}\n")
            trial_output.write(f"Answer: {answer}\n")
            trial_output.write(f"Exact Match: {metric_scores['exact_match']:.4f}\n")
            trial_output.write(f"F1: {metric_scores['f1']:.4f}\n")
            trial_output.write(f"Success: {successful}\n")
            trial_output.write("="*80 + "\n\n")
            
            # Check if answer is different from baseline, if different write to another file
            if successful and answer != baseline_answers[sample_idx]:
                different_answers.write(f"Sample ID: {sample_idx}\n")
                different_answers.write(f"Question: {dataset[sample_idx]['question']}\n")
                different_answers.write(f"Layer: {layer_idx}, Module: {selected_module}, Bit: {bit_position}\n")
                different_answers.write(f"Reference: {reference}\n")
                different_answers.write(f"Baseline: {baseline_answers[sample_idx]}\n")
                different_answers.write(f"Bit-flip: {answer}\n")
                different_answers.write(f"Exact Match: {metric_scores['exact_match']:.4f}\n")
                different_answers.write(f"F1: {metric_scores['f1']:.4f}\n")
                different_answers.write("="*80 + "\n\n")

            # Clean up memory
            del answer
            del reference
            del context
            del question_id
            del answer_start
            torch.cuda.empty_cache()

            bit_flip_progress.update(1)

            # Print intermediate results occasionally
            if bit_flip_progress.n % 100 == 0:
                successful_trials = [t for t in results["bit_flip_trials"] if t["successful"]]
                if successful_trials:
                    avg_exact_match = sum(t["exact_match"] for t in successful_trials) / len(successful_trials)
                    avg_f1 = sum(t["f1"] for t in successful_trials) / len(successful_trials)
                    print(f"\nInterim average Exact Match after {bit_flip_progress.n} trials: {avg_exact_match:.4f}")
                    print(f"Interim average F1 after {bit_flip_progress.n} trials: {avg_f1:.4f}")
        
        # Close current trial output file
        trial_output.close()
        
        # Restore weight or remove hook
        if args.fault_mode == 'weight':
            with torch.no_grad():
                weight_tensor[x, y] = perform_bit_flip_weight(weight_tensor[x, y], bit_position)
        else:
            hook_handle.remove()
            
        # Clean up memory after each trial
        torch.cuda.empty_cache()

    # Close files
    different_answers.close()

    # Compute and print final results
    successful_trials = [t for t in results["bit_flip_trials"] if t["successful"]]
    success_rate = len(successful_trials) / len(results["bit_flip_trials"]) * 100

    avg_exact_match = 0.0
    avg_f1 = 0.0
    if successful_trials:
        avg_exact_match = sum(t["exact_match"] for t in successful_trials) / len(successful_trials)
        avg_f1 = sum(t["f1"] for t in successful_trials) / len(successful_trials)

    avg_time = sum(t["time_taken"] for t in results["bit_flip_trials"]) / len(results["bit_flip_trials"])

    print("\n--- Final Results ---")
    print(f"Baseline average Exact Match: {avg_baseline_exact_match:.4f}")
    print(f"Baseline average F1: {avg_baseline_f1:.4f}")
    print(f"Bit-flip success rate: {success_rate:.2f}%")
    print(f"Average bit-flip Exact Match: {avg_exact_match:.4f}")
    print(f"Average bit-flip F1: {avg_f1:.4f}")
    print(f"Average evaluation time: {avg_time:.2f} seconds")
    print(f"Baseline answers saved to: {baseline_file}")
    print(f"Different answers saved to: {different_answers_file}")
    print(f"Individual bit-flip trial results saved to: {output_dir}/bit_flip_trial_*.txt")

    # Compute impact by module type
    module_impacts = {}
    for module in layer_weights.keys():
        module_trials = [t for t in results["bit_flip_trials"] if t["module"] == module and t["successful"]]
        if module_trials:
            module_exact_match = sum(t["exact_match"] for t in module_trials) / len(module_trials)
            module_f1 = sum(t["f1"] for t in module_trials) / len(module_trials)
            module_impacts[module] = {
                "exact_match": module_exact_match,
                "f1": module_f1
            }

    print("\n--- Impact by Module Type ---")
    for module, scores in module_impacts.items():
        print(f"{module}:")
        print(f"  Exact Match: {scores['exact_match']:.4f} (Δ from baseline: {scores['exact_match'] - avg_baseline_exact_match:.4f})")
        print(f"  F1: {scores['f1']:.4f} (Δ from baseline: {scores['f1'] - avg_baseline_f1:.4f})")

    # Save results to file
    import json
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main() 