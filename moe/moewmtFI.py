import os
import random
import numpy as np
import torch
import time
import struct
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from torch.utils.data import DataLoader
import evaluate
import argparse

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


def perform_bit_flip(tensor, bit_position):
    """Perform bit flip on a tensor value at the specified position."""
#    print(f"\n=== Bit Flip Operation Details ===")
#    print(f"Input tensor value: {tensor.item()}")
    
    with torch.no_grad():
        tensor_bf16 = tensor.to(torch.bfloat16)
        #print(f"After converting to bfloat16: {tensor_bf16.item()}")
        
        bits = tensor_bf16.view(torch.int16)
        bits_value = bits.item()
        #print(f"Bits representation: {bits_value & 0xFFFF:016b}")
        
        mask = (1 << bit_position[0]) | (1 << bit_position[1])
        #print(f"Bit positions to flip: {bit_position}")
        #print(f"Mask: {mask & 0xFFFF:016b}")    
        
        bits = bits ^ mask
        #print(f"After bit flip: {bits.item() & 0xFFFF:016b}") 
        
        flipped_tensor = bits.view(torch.bfloat16)
        #print(f"Final flipped tensor value: {flipped_tensor.item()}")
        #print("==============================\n")
    return flipped_tensor 


def tokenize_function(examples):
    # Format for translation: "Translate from German to English: [German text]"
    texts = [f"translate English to German:{src}"
             for src in examples["de"]]
    result = tokenizer(texts, padding='max_length', truncation=True, max_length=1024)
    result["valid_length"] = [len(x) for x in tokenizer.batch_encode_plus(texts)["input_ids"]]

    return result


def get_input(dataset, tokenizer, id, device):
    german_text = dataset["translation"][id]["de"]
    english_text = dataset["translation"][id]["en"]
    prompt = f"Translate this from German to English:\nGerman: {german_text}\nEnglish:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device, non_blocking=True)
    return english_text, prompt, input_ids


def generate(id, dataset, tokenizer, model, max_length=100):
    """Generate translation for a single example."""
    reference, prompt, input_ids = get_input(dataset, tokenizer, id, device)
    prompt_len = len(input_ids[0])
    source = dataset["translation"][id]["de"]  

    with torch.no_grad():
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

    generated_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    
    # 清理内存
    del output
    del input_ids
    torch.cuda.empty_cache()
    
    return generated_text, reference, source


def compute_bleu_score(prediction, reference):
    """Compute BLEU score for a translation."""
    if not prediction or not reference:
        return 0.0
    bleu = evaluate.load("bleu", experiment_id=f"bleu_{time.time()}")
    results = bleu.compute(predictions=[prediction], references=[[reference]])
    del bleu
    torch.cuda.empty_cache()
    return results["bleu"]


def compute_comet_score(prediction, reference, source):
    """Compute COMET score for a translation."""
    if not prediction or not reference:
        return 0.0
    comet = evaluate.load("comet", experiment_id=f"comet_{time.time()}")
    results = comet.compute(predictions=[prediction], references=[reference], sources=[source])
    del comet
    torch.cuda.empty_cache()
    return results["mean_score"]


def compute_chrf_score(prediction, reference, word_order=2):
    """Compute chrF++ score for a translation."""
    if not prediction or not reference:
        return 0.0
    chrf = evaluate.load("chrf", experiment_id=f"chrf_{time.time()}")
    results = chrf.compute(predictions=[prediction], references=[[reference]], word_order=word_order)
    del chrf
    torch.cuda.empty_cache()
    return results["score"]


def read_qids_from_file(filename):
    """Read sample IDs from file."""
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=500, help='Number of bit flip trials per sample')
    args = parser.parse_args()
    num_trials = args.num_trials  # Just one bit flip trial per sample to keep total experiments manageable

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Create output directory if it does not exist
    output_dir = "moewmtFI"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output files
    baseline_file = os.path.join(output_dir, "baseline_translations.txt")
    different_translations_file = os.path.join(output_dir, "different_translations.txt")
    
    # Open files for writing
    baseline_output = open(baseline_file, "w", encoding="utf-8")
    different_translations = open(different_translations_file, "w", encoding="utf-8")

    # Randomly select 100 input samples
    num_samples = 100

    # Create random sample IDs within the dataset size (assuming it's large enough)
    max_id = 2900  # Assuming Opus-100 validation set has at least 5000 samples
    sample_ids = random.sample(range(max_id), num_samples)

    # Write sample IDs to file for reproducibility
    with open("wmt16_indices_beam.txt", "w") as f:
        for i in sample_ids:
            f.write(f"{i}\n")
    print(evaluate.list_evaluation_modules())
    # You can also read sample IDs from file if needed
    # sample_ids = read_qids_from_file("opus100_indices.txt")

    # Load Opus-100 dataset
    dataset = datasets.load_dataset("wmt/wmt16", "de-en", split="test")
    dataset = dataset.select(sample_ids)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        "DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.resize_token_embeddings(len(tokenizer))
    print("Model and tokenizer loaded")
    for layer in model.children():
        print(layer)
    # Define layer weights for selecting modules to inject errors
    layer_weights = {
        'self_attn.v_proj': 1,
        'self_attn.k_proj': 1,
        'self_attn.q_proj': 3,
        'self_attn.o_proj': 3,
        'block_sparse_moe.experts.0.w1': 8,  # MOE experts weights
        'block_sparse_moe.experts.0.w2': 8,
        'block_sparse_moe.experts.0.w3': 8,
        'block_sparse_moe.experts.1.w1': 8,
        'block_sparse_moe.experts.1.w2': 8,
        'block_sparse_moe.experts.1.w3': 8,
        'block_sparse_moe.experts.2.w1': 8,
        'block_sparse_moe.experts.2.w2': 8,
        'block_sparse_moe.experts.2.w3': 8,
        'block_sparse_moe.experts.3.w1': 8,
        'block_sparse_moe.experts.3.w2': 8,
        'block_sparse_moe.experts.3.w3': 8,
        'block_sparse_moe.experts.4.w1': 8,
        'block_sparse_moe.experts.4.w2': 8,
        'block_sparse_moe.experts.4.w3': 8,
        'block_sparse_moe.experts.5.w1': 8,
        'block_sparse_moe.experts.5.w2': 8,
        'block_sparse_moe.experts.5.w3': 8,
        'block_sparse_moe.experts.6.w1': 8,
        'block_sparse_moe.experts.6.w2': 8,
        'block_sparse_moe.experts.6.w3': 8,
        'block_sparse_moe.experts.7.w1': 8,
        'block_sparse_moe.experts.7.w2': 8,
        'block_sparse_moe.experts.7.w3': 8
        #'mlp.up_proj': 64,
        #'mlp.gate_proj': 64,
        #'mlp.down_proj': 64
    }

    # Calculate total weight for weighted random selection
    total_weight = sum(layer_weights.values())
    layers = list(layer_weights.keys())
    weights = [layer_weights[layer] / total_weight for layer in layers]

    # Create results dictionary
    results = {
        "baseline_translations": [],
        "bit_flip_trials": [],
    }

    # Store baseline translations for later comparison
    baseline_translations = {}

    # Perform baseline translations first (no bit flips)
    print("Generating baseline translations...")
    baseline_progress = tqdm(range(len(dataset)), desc="Baseline translations")

    total_baseline_bleu = 0.0
    #total_baseline_comet = 0.0
    total_baseline_chrf = 0.0
    for idx in range(len(dataset)):
        translation, reference, source = generate(idx, dataset, tokenizer, model)
        bleu_score = compute_bleu_score(translation, reference)
        #comet_score = compute_comet_score(translation, reference, source)
        chrf_score = compute_chrf_score(translation, reference, word_order=2)
        total_baseline_bleu += bleu_score
        #total_baseline_comet += comet_score
        total_baseline_chrf += chrf_score

        results["baseline_translations"].append({
            "sample_id": idx,
            "reference": reference,
            "translation": translation,
            "bleu": bleu_score,
            #"comet": comet_score,
            "chrf": chrf_score
        })
        
        # Write baseline translations to file
        baseline_output.write(f"Sample ID: {idx} (Baseline)\n")
        baseline_output.write(f"Reference: {reference}\n")
        baseline_output.write(f"Translation: {translation}\n")
        baseline_output.write(f"BLEU: {bleu_score:.4f}\n")
        #baseline_output.write(f"COMET: {comet_score:.4f}\n")
        baseline_output.write(f"chrF++: {chrf_score:.4f}\n")
        baseline_output.write("="*80 + "\n\n")
        
        # Store baseline translations for later comparison
        baseline_translations[idx] = translation

        baseline_progress.update(1)

    avg_baseline_bleu = total_baseline_bleu / len(dataset)
    #avg_baseline_comet = total_baseline_comet / len(dataset)
    avg_baseline_chrf = total_baseline_chrf / len(dataset)
    print(f"Baseline average BLEU score: {avg_baseline_bleu:.4f}")
    #print(f"Baseline average COMET score: {avg_baseline_comet:.4f}")
    print(f"Baseline average chrF++ score: {avg_baseline_chrf:.4f}")

    # Close baseline translations file
    baseline_output.close()

    # Perform bit flip experiments - use one trial per sample since we have 100 samples
    print(f"Performing {num_trials} bit flip trials per sample...")

    bit_flip_progress = tqdm(total=len(dataset) * num_trials, desc="Bit flip trials")

    for trial in range(num_trials):
        # Select a random layer index (0-27)
        layer_idx = random.randint(0, 27)

        # Select a module based on weighted random selection
        selected_module = random.choices(layers, weights=weights)[0]

        # Navigate to the target layer and module
        target_layer = model.model.layers[layer_idx]

        module_path = selected_module.split('.')
        current_module = target_layer
        for path_part in module_path:
            current_module = getattr(current_module, path_part)
            
        # Get the weight tensor
        weight_tensor = current_module.weight

        # Select random coordinates in the weight tensor
        x = random.randint(0, weight_tensor.shape[0] - 1)
        y = random.randint(0, weight_tensor.shape[1] - 1)

        # Select a random bit position to flip (0-15 for BF16)
        bit_position = random.sample(range(16), 2)
        # Store original weight value
        original_weight_value = weight_tensor[x, y].clone()

        # Create output file for current trial
        trial_file = os.path.join(output_dir, f"bit_flip_trial_{trial}_layer_{layer_idx}_module_{selected_module.replace('.', '_')}_bit_{bit_position}.txt")
        trial_output = open(trial_file, "w", encoding="utf-8")
        
        # Write details of current trial
        trial_output.write(f"Trial: {trial}\n")
        trial_output.write(f"Layer: {layer_idx}\n")
        trial_output.write(f"Module: {selected_module}\n")
        trial_output.write(f"Bit Position: {bit_position}\n\n")

        # Perform the bit flip
        with torch.no_grad():
            weight_tensor[x, y] = perform_bit_flip(weight_tensor[x, y], bit_position)

        # Generate translation with bit flip
        for sample_idx in range(len(dataset)):
            start_time = time.time()
            translation, reference, source = generate(sample_idx, dataset, tokenizer, model)
            successful = True
            end_time = time.time()

            # Calculate scores
            bleu_score = compute_bleu_score(translation, reference)
            #comet_score = compute_comet_score(translation, reference, source)
            chrf_score = compute_chrf_score(translation, reference)

            # Record results
            results["bit_flip_trials"].append({
                "sample_id": sample_idx,
                "layer_idx": layer_idx,
                "module": selected_module,
                "bit_position": bit_position,
                "reference": reference,
                "translation": translation,
                "bleu": bleu_score,
                #"comet": comet_score,
                "chrf": chrf_score,
                "time_taken": end_time - start_time,
                "successful": successful
            })
            
            # Write to current trial output file
            trial_output.write(f"Sample ID: {sample_idx}\n")
            trial_output.write(f"Reference: {reference}\n")
            trial_output.write(f"Translation: {translation}\n")
            trial_output.write(f"BLEU: {bleu_score:.4f}\n")
            #trial_output.write(f"COMET: {comet_score:.4f}\n")
            trial_output.write(f"chrF++: {chrf_score:.4f}\n")
            trial_output.write(f"Success: {successful}\n")
            trial_output.write("="*80 + "\n\n")
            
            # Check if translation is different from baseline, if so write to another file
            if successful and translation != baseline_translations[sample_idx]:
                different_translations.write(f"Sample ID: {sample_idx}\n")
                different_translations.write(f"Layer: {layer_idx}, Module: {selected_module}, Bit: {bit_position}\n")
                different_translations.write(f"Reference: {reference}\n")
                different_translations.write(f"Baseline: {baseline_translations[sample_idx]}\n")
                different_translations.write(f"Bit-flip: {translation}\n")
                different_translations.write(f"BLEU: {bleu_score:.4f}\n")
                #different_translations.write(f"COMET: {comet_score:.4f}\n")
                different_translations.write(f"chrF++: {chrf_score:.4f}\n")
                different_translations.write("="*80 + "\n\n")

            # Clear memory
            del translation
            del reference
            del source
            torch.cuda.empty_cache()

            bit_flip_progress.update(1)

            # Print interim results occasionally
            if bit_flip_progress.n % 100 == 0:
                successful_trials = [t for t in results["bit_flip_trials"] if t["successful"]]
                if successful_trials:
                    avg_bleu = sum(t["bleu"] for t in successful_trials) / len(successful_trials)
                    #avg_comet = sum(t["comet"] for t in successful_trials) / len(successful_trials)
                    avg_chrf = sum(t["chrf"] for t in successful_trials) / len(successful_trials)
                    print(f"\nInterim average BLEU after {bit_flip_progress.n} trials: {avg_bleu:.4f}")
                    #print(f"Interim average COMET after {bit_flip_progress.n} trials: {avg_comet:.4f}")
                    print(f"Interim average chrF++ after {bit_flip_progress.n} trials: {avg_chrf:.4f}")
        
        # Close current trial output file
        trial_output.close()
            
        with torch.no_grad():
            weight_tensor[x, y] = perform_bit_flip(weight_tensor[x, y], bit_position)
            
        # Clear memory after each trial
        torch.cuda.empty_cache()

    # Close files
    different_translations.close()

    # Calculate and print final results
    successful_trials = [t for t in results["bit_flip_trials"] if t["successful"]]
    success_rate = len(successful_trials) / len(results["bit_flip_trials"]) * 100

    avg_bleu = 0.0
    #avg_comet = 0.0
    avg_chrf = 0.0
    if successful_trials:
        avg_bleu = sum(t["bleu"] for t in successful_trials) / len(successful_trials)
        #avg_comet = sum(t["comet"] for t in successful_trials) / len(successful_trials)
        avg_chrf = sum(t["chrf"] for t in successful_trials) / len(successful_trials)

    avg_time = sum(t["time_taken"] for t in results["bit_flip_trials"]) / len(results["bit_flip_trials"])

    print("\n--- Final Results ---")
    print(f"Baseline average BLEU: {avg_baseline_bleu:.4f}")
    #print(f"Baseline average COMET: {avg_baseline_comet:.4f}")
    print(f"Baseline average chrF++: {avg_baseline_chrf:.4f}")
    print(f"Bit-flip success rate: {success_rate:.2f}%")
    print(f"Average bit-flip BLEU: {avg_bleu:.4f}")
    #print(f"Average bit-flip COMET: {avg_comet:.4f}")
    print(f"Average bit-flip chrF++: {avg_chrf:.4f}")
    print(f"Average evaluation time: {avg_time:.2f} seconds")
    print(f"Baseline translations saved to: {baseline_file}")
    print(f"Different translations saved to: {different_translations_file}")
    print(f"Individual bit-flip trial results saved to: {output_dir}/bit_flip_trial_*.txt")

    # Calculate impact by module type
    module_impacts = {}
    for module in layer_weights.keys():
        module_trials = [t for t in results["bit_flip_trials"] if t["module"] == module and t["successful"]]
        if module_trials:
            module_bleu = sum(t["bleu"] for t in module_trials) / len(module_trials)
            #module_comet = sum(t["comet"] for t in module_trials) / len(module_trials)
            module_chrf = sum(t["chrf"] for t in module_trials) / len(module_trials)
            module_impacts[module] = {
                "bleu": module_bleu,
                #"comet": module_comet,
                "chrf": module_chrf
            }

    print("\n--- Impact by Module Type ---")
    for module, scores in module_impacts.items():
        print(f"{module}:")
        print(f"  BLEU: {scores['bleu']:.4f} (Δ from baseline: {scores['bleu'] - avg_baseline_bleu:.4f})")
        #print(f"  COMET: {scores['comet']:.4f} (Δ from baseline: {scores['comet'] - avg_baseline_comet:.4f})")
        print(f"  chrF++: {scores['chrf']:.4f} (Δ from baseline: {scores['chrf'] - avg_baseline_chrf:.4f})")

    # Save results to file
    import json
    with open("opus_bit_flip_results_beam.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to opus_bit_flip_results.json")


if __name__ == "__main__":
    main()
