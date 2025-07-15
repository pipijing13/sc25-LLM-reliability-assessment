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
            output = output[0]
        
        # Get the current hook call count for this input
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

def record_output_dimensions(model, dataset, tokenizer, device, generation_mode='greedy'):
    """Record baseline evaluation dimension information"""
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
        _, prompt, input_ids = get_input(dataset, tokenizer, idx, device)
        prompt_len = len(input_ids[0])
        
        # Generate output
        with torch.no_grad():
            if generation_mode == 'beam':
                output = model.generate(
                    input_ids,
                    max_length=prompt_len + 80,
                    do_sample=False,
                    num_beams=6,
                    temperature=0.9,
                    no_repeat_ngram_size=3,
                    num_return_sequences=1,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:  # greedy
                output = model.generate(
                    input_ids,
                    max_length=prompt_len + 80,
                    do_sample=False,
                    num_beams=1,
                    temperature=0.9,
                    no_repeat_ngram_size=3,
                    num_return_sequences=1,
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

def tokenize_function(examples):
    # Format input for summarization task
    texts = [f"Summarize the following text:\n{text}" for text in examples["text"]]
    result = tokenizer(texts, padding='max_length', truncation=True, max_length=1024)
    result["valid_length"] = [len(x) for x in tokenizer.batch_encode_plus(texts)["input_ids"]]
    return result

def get_input(dataset, tokenizer, id, device):
    text = dataset[id]["text"]
    summary = dataset[id]["summary"]
    prompt = f"Summarize the following text in less than 30 words:\n{text}\nSummary:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device, non_blocking=True)
    return summary, prompt, input_ids

def generate(id, dataset, tokenizer, model, generation_mode='greedy', max_length=80):
    """Generate summary for a single example."""
    reference, prompt, input_ids = get_input(dataset, tokenizer, id, device)
    prompt_len = len(input_ids[0])
    text = dataset[id]["text"]  # Get original text

    with torch.no_grad():
        if generation_mode == 'beam':
            output = model.generate(
                input_ids,
                max_length=prompt_len + max_length,
                do_sample=False,
                num_beams=6,
                temperature=0.9,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
        else:  # greedy
            output = model.generate(
                input_ids,
                max_length=prompt_len + max_length,
                do_sample=False,
                num_beams=1,
                temperature=0.9,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )

    generated_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    
    # Only take the first sentence
    first_sentence = ""
    # Try using common sentence ending symbols for splitting
    sentence_endings = re.split(r'(?<=[.!?])\s+', generated_text)
    if sentence_endings and len(sentence_endings) > 0:
        first_sentence = sentence_endings[0]
    else:
        first_sentence = generated_text
    
    # Clean up memory
    del output
    del input_ids
    torch.cuda.empty_cache()
    
    return first_sentence, reference, text

def compute_rouge_scores(prediction, reference):
    """Compute ROUGE scores for a summary."""
    if not prediction or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=[prediction], references=[reference])
    del rouge
    torch.cuda.empty_cache()
    return results

def read_qids_from_file(filename):
    """Read sample IDs from file."""
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file]

def main():
    parser = argparse.ArgumentParser(description="XLSum Fault Injection Experiment")
    parser.add_argument('--fault_mode', type=str, default='weight', 
                       choices=['weight', 'neuron', 'single'], 
                       help='Fault injection mode: weight (weight), neuron (neuron output double bit), single (neuron output single bit)')
    parser.add_argument('--generation_mode', type=str, default='greedy',
                       choices=['beam', 'greedy'],
                       help='Generation mode: beam search or greedy decoding')
    parser.add_argument('--model', type=str, default='summarizer',
                       choices=['summarizer', 'llama3', 'qwen'],
                       help='Model type: Meta-Llama-3.1-8B-Instruct-Summarizer, Llama-3.1-8B, or Qwen2.5-7B')
    parser.add_argument('--num_trials', type=int, default=500,
                       help='Number of bit flip trials per sample')
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Create output directory
    output_dir = f"xlsumFI_{args.model}_{args.fault_mode}_{args.generation_mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output files
    baseline_file = os.path.join(output_dir, "baseline_summaries.txt")
    different_summaries_file = os.path.join(output_dir, "different_summaries.txt")
    
    # Open files for writing
    baseline_output = open(baseline_file, "w", encoding="utf-8")
    different_summaries = open(different_summaries_file, "w", encoding="utf-8")

    # Randomly select samples
    num_samples = 100

    # Load XLSum English dataset
    print("Loading XLSum dataset...")
    dataset_full = datasets.load_dataset("csebuetnlp/xlsum", "english", split="test")
    print(f"XLSum English dataset loaded with {len(dataset_full)} samples")
    
    # Create random sample IDs
    max_id = len(dataset_full)
    sample_ids = random.sample(range(max_id), num_samples)

    # Save sample IDs for reproducibility
    with open(os.path.join(output_dir, "sample_indices.txt"), "w") as f:
        for i in sample_ids:
            f.write(f"{i}\n")
    print(evaluate.list_evaluation_modules())

    # Select samples
    dataset = dataset_full.select(sample_ids)
    print(f"Selected {len(dataset)} samples for experiments")

    # Load model and tokenizer based on selected model type
    print("Loading model and tokenizer...")
    global tokenizer
    
    if args.model == 'summarizer':
        model_name = "raaec/Meta-Llama-3.1-8B-Instruct-Summarizer"
        layer_weights = {
            'self_attn.v_proj': 1,
            'self_attn.k_proj': 1,
            'self_attn.q_proj': 4,
            'self_attn.o_proj': 4,
            'mlp.up_proj': 14,
            'mlp.gate_proj': 14,
            'mlp.down_proj': 14
        }
    elif args.model == 'llama3':
        model_name = "meta-llama/Llama-3.1-8B"
        layer_weights = {
            'self_attn.v_proj': 1,
            'self_attn.k_proj': 1,
            'self_attn.q_proj': 4,
            'self_attn.o_proj': 4,
            'mlp.up_proj': 14,
            'mlp.gate_proj': 14,
            'mlp.down_proj': 14
        }
    else:  # qwen
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        layer_weights = {
            'self_attn.v_proj': 1,
            'self_attn.k_proj': 1,
            'self_attn.q_proj': 7,
            'self_attn.o_proj': 7,
            'mlp.up_proj': 37,
            'mlp.gate_proj': 37,
            'mlp.down_proj': 37
        }

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
    print(f"Model {args.model} loaded")
    for layer in model.children():
        print(layer)

    # Calculate total weight for weighted random selection
    total_weight = sum(layer_weights.values())
    layers = list(layer_weights.keys())
    weights = [layer_weights[layer] / total_weight for layer in layers]

    # Create results dictionary
    results = {
        "baseline_summaries": [],
        "bit_flip_trials": [],
    }

    # Store baseline summaries for comparison
    baseline_summaries = {}

    # Generate baseline summaries
    print("Generating baseline summaries...")
    baseline_progress = tqdm(range(len(dataset)), desc="Baseline summaries")

    total_baseline_rouge1 = 0.0
    total_baseline_rouge2 = 0.0
    total_baseline_rougeL = 0.0
    
    for idx in range(len(dataset)):
        summary, reference, text = generate(idx, dataset, tokenizer, model, args.generation_mode)
        rouge_scores = compute_rouge_scores(summary, reference)
        
        total_baseline_rouge1 += rouge_scores["rouge1"]
        total_baseline_rouge2 += rouge_scores["rouge2"]
        total_baseline_rougeL += rouge_scores["rougeL"]

        results["baseline_summaries"].append({
            "sample_id": idx,
            "reference": reference,
            "summary": summary,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"]
        })
        
        # Write baseline summary to file
        baseline_output.write(f"Sample ID: {idx} (Baseline)\n")
        baseline_output.write(f"Reference: {reference}\n")
        baseline_output.write(f"Summary: {summary}\n")
        baseline_output.write(f"ROUGE-1: {rouge_scores['rouge1']:.4f}\n")
        baseline_output.write(f"ROUGE-2: {rouge_scores['rouge2']:.4f}\n")
        baseline_output.write(f"ROUGE-L: {rouge_scores['rougeL']:.4f}\n")
        baseline_output.write("="*80 + "\n\n")
        
        # Store baseline summary for comparison
        baseline_summaries[idx] = summary

        baseline_progress.update(1)

    avg_baseline_rouge1 = total_baseline_rouge1 / len(dataset)
    avg_baseline_rouge2 = total_baseline_rouge2 / len(dataset)
    avg_baseline_rougeL = total_baseline_rougeL / len(dataset)
    print(f"Baseline average ROUGE-1 score: {avg_baseline_rouge1:.4f}")
    print(f"Baseline average ROUGE-2 score: {avg_baseline_rouge2:.4f}")
    print(f"Baseline average ROUGE-L score: {avg_baseline_rougeL:.4f}")

    # Close baseline file
    baseline_output.close()
    
    if args.fault_mode in ["neuron", "single"]:
        min_seq_len, min_tokens = record_output_dimensions(model, dataset, tokenizer, device, args.generation_mode)

    # Perform bit flip experiments
    print(f"Performing {args.num_trials} bit flip trials...")
    bit_flip_progress = tqdm(total=len(dataset) * args.num_trials, desc="Bit flip trials")

    for trial in range(args.num_trials):
        # Select random layer index
        layer_idx = random.randint(0, 27)

        # Select module based on weights
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
        else:
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
        
        # Write trial information
        trial_output.write(f"Trial: {trial}\n")
        trial_output.write(f"Layer: {layer_idx}\n")
        trial_output.write(f"Module: {selected_module}\n")
        trial_output.write(f"Bit Position: {bit_position}\n\n")

        # Generate summaries with bit flips
        for sample_idx in range(len(dataset)):
            if args.fault_mode in ["neuron", "single"] and hasattr(hook, 'count'):
                delattr(hook, 'count')
                
            start_time = time.time()
            summary, reference, text = generate(sample_idx, dataset, tokenizer, model, args.generation_mode)
            successful = True
            end_time = time.time()

            # Compute scores
            rouge_scores = compute_rouge_scores(summary, reference)

            # Record results
            trial_result = {
                "sample_id": sample_idx,
                "layer_idx": layer_idx,
                "module": selected_module,
                "bit_position": bit_position,
                "reference": reference,
                "summary": summary,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "time_taken": end_time - start_time,
                "successful": successful
            }
            
            if args.fault_mode == 'weight':
                trial_result["original_weight_value"] = original_weight_value.item()
                trial_result["flipped_weight_value"] = weight_tensor[x, y].item()
            
            results["bit_flip_trials"].append(trial_result)
            
            # Write to current trial output file
            trial_output.write(f"Sample ID: {sample_idx}\n")
            trial_output.write(f"Reference: {reference}\n")
            trial_output.write(f"Summary: {summary}\n")
            trial_output.write(f"ROUGE-1: {rouge_scores['rouge1']:.4f}\n")
            trial_output.write(f"ROUGE-2: {rouge_scores['rouge2']:.4f}\n")
            trial_output.write(f"ROUGE-L: {rouge_scores['rougeL']:.4f}\n")
            trial_output.write(f"Success: {successful}\n")
            trial_output.write("="*80 + "\n\n")
            
            # If summary is different from baseline, write to different_summaries file
            if successful and summary != baseline_summaries[sample_idx]:
                different_summaries.write(f"Sample ID: {sample_idx}\n")
                different_summaries.write(f"Layer: {layer_idx}, Module: {selected_module}, Bit: {bit_position}\n")
                different_summaries.write(f"Reference: {reference}\n")
                different_summaries.write(f"Baseline: {baseline_summaries[sample_idx]}\n")
                different_summaries.write(f"Bit-flip: {summary}\n")
                different_summaries.write(f"ROUGE-1: {rouge_scores['rouge1']:.4f}\n")
                different_summaries.write(f"ROUGE-2: {rouge_scores['rouge2']:.4f}\n")
                different_summaries.write(f"ROUGE-L: {rouge_scores['rougeL']:.4f}\n")
                different_summaries.write("="*80 + "\n\n")

            # Clean up memory
            del summary
            del reference
            del text
            torch.cuda.empty_cache()

            bit_flip_progress.update(1)

            # Print intermediate results periodically
            if bit_flip_progress.n % 100 == 0:
                successful_trials = [t for t in results["bit_flip_trials"] if t["successful"]]
                if successful_trials:
                    avg_rouge1 = sum(t["rouge1"] for t in successful_trials) / len(successful_trials)
                    avg_rouge2 = sum(t["rouge2"] for t in successful_trials) / len(successful_trials)
                    avg_rougeL = sum(t["rougeL"] for t in successful_trials) / len(successful_trials)
                    print(f"\nInterim average ROUGE-1 after {bit_flip_progress.n} trials: {avg_rouge1:.4f}")
                    print(f"Interim average ROUGE-2 after {bit_flip_progress.n} trials: {avg_rouge2:.4f}")
                    print(f"Interim average ROUGE-L after {bit_flip_progress.n} trials: {avg_rougeL:.4f}")
        
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
    different_summaries.close()

    # Calculate and print final results
    successful_trials = [t for t in results["bit_flip_trials"] if t["successful"]]
    success_rate = len(successful_trials) / len(results["bit_flip_trials"]) * 100

    avg_rouge1 = 0.0
    avg_rouge2 = 0.0
    avg_rougeL = 0.0
    if successful_trials:
        avg_rouge1 = sum(t["rouge1"] for t in successful_trials) / len(successful_trials)
        avg_rouge2 = sum(t["rouge2"] for t in successful_trials) / len(successful_trials)
        avg_rougeL = sum(t["rougeL"] for t in successful_trials) / len(successful_trials)

    avg_time = sum(t["time_taken"] for t in results["bit_flip_trials"]) / len(results["bit_flip_trials"])

    print("\n--- Final Results ---")
    print(f"Baseline average ROUGE-1: {avg_baseline_rouge1:.4f}")
    print(f"Baseline average ROUGE-2: {avg_baseline_rouge2:.4f}")
    print(f"Baseline average ROUGE-L: {avg_baseline_rougeL:.4f}")
    print(f"Bit-flip success rate: {success_rate:.2f}%")
    print(f"Average bit-flip ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average bit-flip ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average bit-flip ROUGE-L: {avg_rougeL:.4f}")
    print(f"Average evaluation time: {avg_time:.2f} seconds")
    print(f"Baseline summaries saved to: {baseline_file}")
    print(f"Different summaries saved to: {different_summaries_file}")
    print(f"Individual bit-flip trial results saved to: {output_dir}/bit_flip_trial_*.txt")

    # Calculate impact by module type
    module_impacts = {}
    for module in layer_weights.keys():
        module_trials = [t for t in results["bit_flip_trials"] if t["module"] == module and t["successful"]]
        if module_trials:
            module_rouge1 = sum(t["rouge1"] for t in module_trials) / len(module_trials)
            module_rouge2 = sum(t["rouge2"] for t in module_trials) / len(module_trials)
            module_rougeL = sum(t["rougeL"] for t in module_trials) / len(module_trials)
            module_impacts[module] = {
                "rouge1": module_rouge1,
                "rouge2": module_rouge2,
                "rougeL": module_rougeL
            }

    print("\n--- Impact by Module Type ---")
    for module, scores in module_impacts.items():
        print(f"{module}:")
        print(f"  ROUGE-1: {scores['rouge1']:.4f} (Δ from baseline: {scores['rouge1'] - avg_baseline_rouge1:.4f})")
        print(f"  ROUGE-2: {scores['rouge2']:.4f} (Δ from baseline: {scores['rouge2'] - avg_baseline_rouge2:.4f})")
        print(f"  ROUGE-L: {scores['rougeL']:.4f} (Δ from baseline: {scores['rougeL'] - avg_baseline_rougeL:.4f})")

    # Save results to file
    import json
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main() 