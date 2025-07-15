import os
import random
import numpy as np
import torch
import time
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import argparse
import evaluate

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

def get_input(dataset, tokenizer, id, device):
    german_text = dataset["translation"][id]["de"]
    english_text = dataset["translation"][id]["en"]
    prompt = f"Translate this from German to English:\nGerman: {german_text}\nEnglish:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device, non_blocking=True)
    return english_text, prompt, input_ids, german_text

def generate(id, dataset, tokenizer, model, generation_mode='greedy', max_length=100):
    """Generate translation for a single example."""
    reference, prompt, input_ids, source = get_input(dataset, tokenizer, id, device)
    prompt_len = len(input_ids[0])

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
    
    # Clean up memory
    del output
    del input_ids
    torch.cuda.empty_cache()
    
    return generated_text, reference, source

def compute_bleu_score(prediction, reference):
    """Compute BLEU score for a translation."""
    if not prediction or not reference:
        return 0.0
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=[prediction], references=[[reference]])
    del bleu
    torch.cuda.empty_cache()
    return results["bleu"]

def compute_chrf_score(prediction, reference, word_order=2):
    """Compute chrF++ score for a translation."""
    if not prediction or not reference:
        return 0.0
    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=[prediction], references=[[reference]], word_order=word_order)
    del chrf
    torch.cuda.empty_cache()
    return results["score"]

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
    """记录baseline评估时的维度信息"""
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
        _, prompt, input_ids, _ = get_input(dataset, tokenizer, idx, device)
        prompt_len = len(input_ids[0])
        
        # Generate output
        with torch.no_grad():
            if generation_mode == 'beam':
                output = model.generate(
                    input_ids,
                    max_length=prompt_len + 100,
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
                    max_length=prompt_len + 100,
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

def main():
    parser = argparse.ArgumentParser(description="WMT16 Fault Injection Experiment")
    parser.add_argument('--fault_mode', type=str, default='weight', 
                       choices=['weight', 'neuron', 'single'], 
                       help='Fault injection mode: weight (weight), neuron (neuron output double bit), single (neuron output single bit)')
    parser.add_argument('--generation_mode', type=str, default='greedy',
                       choices=['beam', 'greedy'],
                       help='Generation mode: beam search or greedy decoding')
    parser.add_argument('--model', type=str, default='alma',
                       choices=['alma', 'qwen', 'llama2'],
                       help='Model type: ALMA-7B, Qwen-7B, or LLaMA2-7B')
    parser.add_argument('--num_trials', type=int, default=500,
                       help='Number of bit flip trials per sample')
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Create output directory
    output_dir = f"wmt16FI_{args.model}_{args.fault_mode}_{args.generation_mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file
    baseline_file = os.path.join(output_dir, "baseline_translations.txt")
    different_translations_file = os.path.join(output_dir, "different_translations.txt")
    
    # Open file for writing
    baseline_output = open(baseline_file, "w", encoding="utf-8")
    different_translations = open(different_translations_file, "w", encoding="utf-8")

    # Randomly select 100 samples
    num_samples = 100
    max_id = 2900
    sample_ids = random.sample(range(max_id), num_samples)

    # Save sample IDs for reproducibility
    with open(os.path.join(output_dir, "sample_indices.txt"), "w") as f:
        for i in sample_ids:
            f.write(f"{i}\n")

    # Load dataset
    dataset = datasets.load_dataset("wmt/wmt16", "de-en", split="test")
    dataset = dataset.select(sample_ids)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    global tokenizer

    # Load model and tokenizer based on selected model type
    if args.model == 'alma':
        model_name = "haoranxu/ALMA-7B"
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
            'self_attn.v_proj': 16,
            'self_attn.k_proj': 16,
            'self_attn.q_proj': 16,
            'self_attn.o_proj': 16,
            'mlp.up_proj': 43,
            'mlp.gate_proj': 43,
            'mlp.down_proj': 43
        }
    elif args.model == 'qwen':
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
    else:  # llama2
        model_name = "meta-llama/Llama-2-7b-hf"
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
            'self_attn.v_proj': 16,
            'self_attn.k_proj': 16,
            'self_attn.q_proj': 16,
            'self_attn.o_proj': 16,
            'mlp.up_proj': 43,
            'mlp.gate_proj': 43,
            'mlp.down_proj': 43
        }

    print(f"Model {args.model} loaded")
    print(f"Number of layers: {num_layers}")
    for layer in model.children():
        print(layer)

    # Define layer weights
    total_weight = sum(layer_weights.values())
    layers = list(layer_weights.keys())
    weights = [layer_weights[layer] / total_weight for layer in layers]

    # Create results dictionary
    results = {
        "baseline_translations": [],
        "bit_flip_trials": [],
    }
    baseline_translations = {}

    # Generate baseline translations
    print("Generating baseline translations...")
    baseline_progress = tqdm(range(len(dataset)), desc="Baseline translations")

    total_baseline_bleu = 0.0
    total_baseline_chrf = 0.0
    for idx in range(len(dataset)):
        translation, reference, source = generate(idx, dataset, tokenizer, model, args.generation_mode)
        bleu_score = compute_bleu_score(translation, reference)
        chrf_score = compute_chrf_score(translation, reference)
        total_baseline_bleu += bleu_score
        total_baseline_chrf += chrf_score

        results["baseline_translations"].append({
            "sample_id": idx,
            "reference": reference,
            "translation": translation,
            "bleu": bleu_score,
            "chrf": chrf_score
        })
        
        baseline_output.write(f"Sample ID: {idx} (Baseline)\n")
        baseline_output.write(f"Reference: {reference}\n")
        baseline_output.write(f"Translation: {translation}\n")
        baseline_output.write(f"BLEU: {bleu_score:.4f}\n")
        baseline_output.write(f"chrF++: {chrf_score:.4f}\n")
        baseline_output.write("="*80 + "\n\n")
        
        baseline_translations[idx] = translation
        baseline_progress.update(1)

    avg_baseline_bleu = total_baseline_bleu / len(dataset)
    avg_baseline_chrf = total_baseline_chrf / len(dataset)
    print(f"Baseline average BLEU score: {avg_baseline_bleu:.4f}")
    print(f"Baseline average chrF++ score: {avg_baseline_chrf:.4f}")

    baseline_output.close()
    
    if args.fault_mode in ["neuron", "single"]:
        min_seq_len, min_tokens = record_output_dimensions(model, dataset, tokenizer, device, args.generation_mode)

    # Perform bit flip experiments
    print(f"Performing {args.num_trials} bit flip trials...")
    bit_flip_progress = tqdm(total=len(dataset) * args.num_trials, desc="Bit flip trials")

    for trial in range(args.num_trials):
        layer_idx = random.randint(0, num_layers - 1)  # Use loaded num_layers
        selected_module = random.choices(layers, weights=weights)[0]
        target_layer = model.model.layers[layer_idx]

        module_path = selected_module.split('.')
        current_module = target_layer
        for path_part in module_path:
            current_module = getattr(current_module, path_part)
            
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

        # Translate each sample
        for sample_idx in range(len(dataset)):
            if args.fault_mode in ["neuron", "single"] and hasattr(hook, 'count'):
                delattr(hook, 'count')
                
            start_time = time.time()
            translation, reference, source = generate(sample_idx, dataset, tokenizer, model, args.generation_mode)
            end_time = time.time()

            # Compute scores
            bleu_score = compute_bleu_score(translation, reference)
            chrf_score = compute_chrf_score(translation, reference)

            # Record results
            trial_result = {
                "sample_id": sample_idx,
                "layer_idx": layer_idx,
                "module": selected_module,
                "bit_position": bit_position,
                "reference": reference,
                "translation": translation,
                "bleu": bleu_score,
                "chrf": chrf_score,
                "time_taken": end_time - start_time
            }
            
            if args.fault_mode == 'weight':
                trial_result["original_weight_value"] = original_weight_value.item()
                trial_result["flipped_weight_value"] = weight_tensor[x, y].item()
            
            results["bit_flip_trials"].append(trial_result)
            
            # Write to current trial output file
            trial_output.write(f"Sample ID: {sample_idx}\n")
            trial_output.write(f"Reference: {reference}\n")
            trial_output.write(f"Translation: {translation}\n")
            trial_output.write(f"BLEU: {bleu_score:.4f}\n")
            trial_output.write(f"chrF++: {chrf_score:.4f}\n")
            trial_output.write("="*80 + "\n\n")
            
            # If translation is different from baseline, write to different_translations file
            if translation != baseline_translations[sample_idx]:
                different_translations.write(f"Sample ID: {sample_idx}\n")
                different_translations.write(f"Layer: {layer_idx}, Module: {selected_module}, Bit: {bit_position}\n")
                different_translations.write(f"Reference: {reference}\n")
                different_translations.write(f"Baseline: {baseline_translations[sample_idx]}\n")
                different_translations.write(f"Bit-flip: {translation}\n")
                different_translations.write(f"BLEU: {bleu_score:.4f}\n")
                different_translations.write(f"chrF++: {chrf_score:.4f}\n")
                different_translations.write("="*80 + "\n\n")

            # Clean up memory
            del translation
            del reference
            del source
            torch.cuda.empty_cache()

            bit_flip_progress.update(1)

            # Print intermediate results periodically
            if bit_flip_progress.n % 100 == 0:
                current_trials = results["bit_flip_trials"]
                if current_trials:
                    avg_bleu = sum(t["bleu"] for t in current_trials) / len(current_trials)
                    avg_chrf = sum(t["chrf"] for t in current_trials) / len(current_trials)
                    print(f"\nInterim average BLEU after {bit_flip_progress.n} trials: {avg_bleu:.4f}")
                    print(f"Interim average chrF++ after {bit_flip_progress.n} trials: {avg_chrf:.4f}")
        
        # Close current trial output file
        trial_output.close()
        
        # Restore weight or remove hook
        if args.fault_mode == 'weight':
            with torch.no_grad():
                weight_tensor[x, y] = perform_bit_flip_weight(weight_tensor[x, y], bit_position)
        else:
            hook_handle.remove()
            
        # Clean up memory
        torch.cuda.empty_cache()

    # Close files
    different_translations.close()

    # Compute and print final results
    all_trials = results["bit_flip_trials"]
    avg_bleu = sum(t["bleu"] for t in all_trials) / len(all_trials)
    avg_chrf = sum(t["chrf"] for t in all_trials) / len(all_trials)
    avg_time = sum(t["time_taken"] for t in all_trials) / len(all_trials)

    print("\n--- Final Results ---")
    print(f"Baseline average BLEU: {avg_baseline_bleu:.4f}")
    print(f"Baseline average chrF++: {avg_baseline_chrf:.4f}")
    print(f"Bit-flip average BLEU: {avg_bleu:.4f}")
    print(f"Bit-flip average chrF++: {avg_chrf:.4f}")
    print(f"Average evaluation time: {avg_time:.2f} seconds")
    print(f"Baseline translations saved to: {baseline_file}")
    print(f"Different translations saved to: {different_translations_file}")
    print(f"Individual bit-flip trial results saved to: {output_dir}/bit_flip_trial_*.txt")

    # Compute impact by module type
    module_impacts = {}
    for module in layer_weights.keys():
        module_trials = [t for t in all_trials if t["module"] == module]
        if module_trials:
            module_bleu = sum(t["bleu"] for t in module_trials) / len(module_trials)
            module_chrf = sum(t["chrf"] for t in module_trials) / len(module_trials)
            module_impacts[module] = {
                "bleu": module_bleu,
                "chrf": module_chrf
            }

    print("\n--- Impact by Module Type ---")
    for module, scores in module_impacts.items():
        print(f"{module}:")
        print(f"  BLEU: {scores['bleu']:.4f} (Δ from baseline: {scores['bleu'] - avg_baseline_bleu:.4f})")
        print(f"  chrF++: {scores['chrf']:.4f} (Δ from baseline: {scores['chrf'] - avg_baseline_chrf:.4f})")

    # Save results to file
    import json
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main() 