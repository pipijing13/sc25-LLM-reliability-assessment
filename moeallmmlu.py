import os
import random
import numpy as np
import torch
import time
import struct
import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
import lm_eval
import tinyBenchmarks

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


def save_detailed_results(results, filename="weight_bit_flip_detailed_results.json"):
    """Save detailed results to a JSON file with proper handling of numpy types."""

    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_types(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.dtype):
            #elif isinstance(obj, tuple):
            #return [convert_numpy_types(item) for item in obj]
            return str(obj)
        elif isinstance(obj, tuple):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, torch.dtype):
            return str(obj)
        else:
            return obj

    # Convert the results and save
    converted_results = convert_numpy_types(results)
    #try:
    #    with open(filename, "w") as f:
    #        json.dump(converted_results, f, indent=2)
    #    print(f"\nDetailed results saved to {filename}")
    #except TypeError as e:
        #print(f"Error during JSON serialization: {e}")
        # 打印出无法序列化的对象类型
        #print("Trying to identify problematic objects...")
        #problematic_types = set()
        #def find_problematic_types(obj):
        #    if isinstance(obj, dict):
        #        for k, v in obj.items():
        #            find_problematic_types(v)
        #    elif isinstance(obj, list):
        #        for item in obj:
        #            find_problematic_types(item)
        #    elif not isinstance(obj, (int, float, str, bool, type(None))):
        #        problematic_types.add(type(obj).__name__)

        #find_problematic_types(converted_results)
        #print(f"Potentially problematic types: {problematic_types}")

    with open(filename, "w") as f:
        json.dump(converted_results, f, indent=2)

    print(f"\nDetailed results saved to {filename}")


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    output_dir = "moetinymmlu"
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = lm_eval.models.huggingface.HFLM(pretrained="DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B", dtype=torch.float16,
                                            device_map="cuda")

    layer_weights = {
        'self_attn.v_proj': 1,
        'self_attn.k_proj': 1,
        'self_attn.q_proj': 3,
        'self_attn.o_proj': 3,
        'block_sparse_moe.experts.0.w1': 8,  
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

    total_weight = sum(layer_weights.values())
    layers = list(layer_weights.keys())
    weights = [layer_weights[layer] / total_weight for layer in layers]

    task = "tinyMMLU"  

    results = {
        "base_accuracy": None,
        "base_results_detail": None,  
        "bit_flip_trials": [],
    }

    print("Performing baseline evaluation...")
    baseline_results = evaluator.simple_evaluate(
        model=model,
        tasks=[task],
        batch_size=1,
        num_fewshot=0,
        device=device,
    )

    results["base_accuracy"] = baseline_results["results"][task]["acc_norm,none"]
    results["base_results_detail"] = baseline_results
    print(f"Baseline accuracy on {task}: {results['base_accuracy']:.4f}")

    baseline_results_path = os.path.join(output_dir, "baseline_results.json")
    save_detailed_results(baseline_results, baseline_results_path)

    num_trials = 1000
    progress_bar = tqdm(range(num_trials), desc="Bit flip trials")

    for trial in range(num_trials):
        trial_seed = 196 + trial
        seed_torch(trial_seed)
        layer_idx = random.randint(0, 27)

        selected_module = random.choices(layers, weights=weights)[0]

        target_layer = model.model.model.layers[layer_idx]

        module_path = selected_module.split('.')
        current_module = target_layer
        for path_part in module_path:
            current_module = getattr(current_module, path_part)

        weight_tensor = current_module.weight

        x = random.randint(0, weight_tensor.shape[0] - 1)
        y = random.randint(0, weight_tensor.shape[1] - 1)

        #bit_position = random.randint(0, 15)
        bit_position = random.sample(range(16), 2)
        original_value = weight_tensor[x, y].item()

        with torch.no_grad():
            weight_tensor[x, y] = perform_bit_flip(weight_tensor[x, y], bit_position)

        flipped_value = weight_tensor[x, y].item()

        start_time = time.time()
        bit_flip_results = evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            batch_size=1,
            num_fewshot=0,
            device=device,
        )
        end_time = time.time()

        with torch.no_grad():
            weight_tensor[x, y] = perform_bit_flip(weight_tensor[x, y], bit_position)

        bit_flip_acc = bit_flip_results["results"][task]["acc_norm,none"]

        trial_data = {
            "trial_id": trial,
            "layer_idx": layer_idx,
            "module": selected_module,
            "coordinates": {"x": int(x), "y": int(y)},
            "bit_position": bit_position,
            "original_value": original_value,
            "flipped_value": flipped_value,
            "accuracy": bit_flip_acc,
            "accuracy_change": bit_flip_acc - results["base_accuracy"],
            "time_taken": end_time - start_time,
            "detailed_results": bit_flip_results
        }

        results["bit_flip_trials"].append(trial_data)

        trial_results_path = os.path.join(output_dir, f"trial_{trial:03d}_results.json")
        save_detailed_results(trial_data, trial_results_path)

        if (trial + 1) % 10 == 0:
            avg_acc = np.mean([trial["accuracy"] for trial in results["bit_flip_trials"]])
            print(f"Interim average accuracy after {trial + 1} trials: {avg_acc:.4f}")

            save_detailed_results(results, os.path.join(output_dir, "all_results_interim.json"))

        progress_bar.update(1)

    avg_accuracy = np.mean([trial["accuracy"] for trial in results["bit_flip_trials"]])
    accuracy_stddev = np.std([trial["accuracy"] for trial in results["bit_flip_trials"]])
    avg_time = np.mean([trial["time_taken"] for trial in results["bit_flip_trials"]])

    print("\n--- Final Results ---")
    print(f"Baseline accuracy: {results['base_accuracy']:.4f}")
    print(f"Average bit-flip accuracy: {avg_accuracy:.4f} (± {accuracy_stddev:.4f})")
    print(f"Average evaluation time: {avg_time:.2f} seconds")

    module_impacts = {}
    for module in layer_weights.keys():
        module_trials = [t for t in results["bit_flip_trials"] if t["module"] == module]
        if module_trials:
            module_impact = np.mean([t["accuracy"] for t in module_trials])
            module_impacts[module] = module_impact

    print("\n--- Impact by Module Type ---")
    for module, impact in module_impacts.items():
        print(f"{module}: {impact:.4f} (Δ from baseline: {impact - results['base_accuracy']:.4f})")

    results["summary"] = {
        "avg_accuracy": float(avg_accuracy),
        "accuracy_stddev": float(accuracy_stddev),
        "avg_time": float(avg_time),
        "module_impacts": module_impacts
    }

    save_detailed_results(results, os.path.join(output_dir, "all_results_final.json"))

    summary_results = {
        "base_accuracy": results["base_accuracy"],
        "summary": results["summary"],
        "bit_flip_trials": [{k: v for k, v in trial.items() if k != 'detailed_results'}
                            for trial in results["bit_flip_trials"]]
    }
    save_detailed_results(summary_results, os.path.join(output_dir, "summary_results.json"))

    print("\nAll results saved to the 'bit_flip_results' directory")


if __name__ == "__main__":
    main()
