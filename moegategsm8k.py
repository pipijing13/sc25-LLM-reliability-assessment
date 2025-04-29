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
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.multiprocessing.set_start_method('spawn', force=True)
device = "cuda"

selected_experts_list = []
baseline_experts = None  

def collect_gate_outputs(model):
    global selected_experts_list
    selected_experts_outputs = []
    
    for selected_experts in selected_experts_list:
        selected_experts_outputs.append(selected_experts.detach().cpu().numpy().tolist())

    selected_experts_list = []
    
    return {
        "selected_experts": selected_experts_outputs
    }

def save_all_results(results_dict, filename="all_results.json"):
    """Save all results to a JSON file"""
    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=2)

def seed_torch(seed=196):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def perform_bit_flip(tensor, bit_position):
    """Perform bit flip on a tensor value at the specified position."""
    print(f"\n=== Bit Flip Operation Details ===")
    print(f"Input tensor value: {tensor.item()}")
    
    with torch.no_grad():
        tensor_bf16 = tensor.to(torch.bfloat16)
        bits = tensor_bf16.view(torch.int16)
        bits_value = bits.item()
        mask = (1 << bit_position)
        bits = bits ^ mask
        flipped_tensor = bits.view(torch.bfloat16)
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
        elif isinstance(obj, (np.dtype, torch.dtype)):
            return str(obj)
        elif isinstance(obj, tuple):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    converted_results = convert_numpy_types(results)
    with open(filename, "w") as f:
        json.dump(converted_results, f, indent=2)

    print(f"\nDetailed results saved to {filename}")

def plot_experts_comparison(baseline_experts, trial_experts_list, output_dir, results):
    os.makedirs(os.path.join(output_dir, "expert_plots"), exist_ok=True)
    
    for i, trial_experts in enumerate(trial_experts_list):
        plt.figure(figsize=(12, 6))
        
        plt.plot(baseline_experts[0], label='Baseline', linewidth=2, marker='o', color='blue')
        
        plt.plot(trial_experts[0], label=f'Trial {i}', linewidth=2, marker='x', color='red')

        trial_info = results["bit_flip_trials"][i]
        trial_acc = trial_info["accuracy"]
        
        plt.title(f'Trial {i} Expert Selection Comparison (Accuracy: {trial_acc:.4f})')
        plt.xlabel('Token Position')
        plt.ylabel('Selected Expert')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(output_dir, "expert_plots", f'experts_comparison_trial_{i:03d}.png'))
        plt.close()
        
    plt.figure(figsize=(14, 8))
    
    diff_rates = []
    accuracies = []
    trial_ids = []
    
    for i, trial_experts in enumerate(trial_experts_list):
        baseline_array = np.array(baseline_experts[0])
        trial_array = np.array(trial_experts[0])
        diff_count = np.sum(baseline_array != trial_array)
        diff_rate = diff_count / len(baseline_array) * 100 
        
        diff_rates.append(diff_rate)
        trial_info = results["bit_flip_trials"][i]
        accuracies.append(trial_info["accuracy"])
        trial_ids.append(i)
    
    plt.scatter(diff_rates, accuracies, c='blue', alpha=0.7)
    
    z = np.polyfit(diff_rates, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(diff_rates, p(diff_rates), "r--", alpha=0.7)
    
    for i, txt in enumerate(trial_ids):
        plt.annotate(txt, (diff_rates[i], accuracies[i]), fontsize=9)
    
    plt.title('Relationship Between Expert Selection Difference and Accuracy')
    plt.xlabel('Expert Selection Difference Rate (%)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, "expert_plots", 'expert_diff_vs_accuracy.png'))
    plt.close()

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    output_dir = "moegategsm8k"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model and tokenizer...")
    model = lm_eval.models.huggingface.HFLM(pretrained="DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B", dtype=torch.bfloat16,
                                            device_map="cuda")

    def hook_fn(module, input, output):
        global selected_experts_list, baseline_experts
        routing_weights = F.softmax(output, dim=1, dtype=torch.float)
        _, selected_experts = torch.topk(routing_weights, 2, dim=-1)
        selected_experts_list.append(selected_experts)
        
        if baseline_experts is None:
            baseline_experts = selected_experts.detach().cpu().numpy()

    for layer in model.model.model.layers:
        layer.block_sparse_moe.gate.register_forward_hook(hook_fn)

    task = "tinyGSM8k"

    results = {
        "base_accuracy": None,
        "base_results_detail": None, 
        "bit_flip_trials": [],
        "selected_experts_baseline": None,  
        "selected_experts_trials": []  
    }

    print("Performing baseline evaluation...")
    baseline_results = evaluator.simple_evaluate(
        model=model,
        tasks=[task],
        batch_size=1,
        num_fewshot=0,
        device=device,
    )

    results["base_accuracy"] = baseline_results["results"][task]["exact_match,flexible-extract"]
    results["base_results_detail"] = baseline_results
    print(f"Baseline accuracy on {task}: {results['base_accuracy']:.4f}")

    gate_outputs_baseline = collect_gate_outputs(model)
    results["selected_experts_baseline"] = gate_outputs_baseline["selected_experts"]

    baseline_results_path = os.path.join(output_dir, "baseline_results.json")
    save_detailed_results(baseline_results, baseline_results_path)

    trial_experts_list = []

    num_trials = 20
    progress_bar = tqdm(range(num_trials), desc="Gate layer bit flip trials")

    for trial in range(num_trials):
        trial_seed = 196 + trial
        seed_torch(trial_seed)
        
        layer_idx = random.randint(0, 27)
        target_layer = model.model.model.layers[layer_idx]
        
        gate_layer = target_layer.block_sparse_moe.gate
        gate_weight = gate_layer.weight
        
        x = random.randint(0, gate_weight.shape[0] - 1)
        y = random.randint(0, gate_weight.shape[1] - 1)
        bit_position = 14  

        original_value = gate_weight[x, y].item()

        with torch.no_grad():
            gate_weight[x, y] = perform_bit_flip(gate_weight[x, y], bit_position)

        flipped_value = gate_weight[x, y].item()
        
        start_time = time.time()
        bit_flip_results = evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            batch_size=1,
            num_fewshot=0,
            device=device,
        )
        end_time = time.time()

        gate_outputs_trial = collect_gate_outputs(model)
        
        if gate_outputs_trial["selected_experts"]:
            trial_experts_list.append(np.array(gate_outputs_trial["selected_experts"][0]))

        with torch.no_grad():
            gate_weight[x, y] = perform_bit_flip(gate_weight[x, y], bit_position)

        bit_flip_acc = bit_flip_results["results"][task]["exact_match,flexible-extract"]

        trial_data = {
            "trial_id": trial,
            "layer_idx": layer_idx,
            "coordinates": {"x": int(x), "y": int(y)},
            "bit_position": bit_position,
            "original_value": original_value,
            "flipped_value": flipped_value,
            "accuracy": bit_flip_acc,
            "accuracy_change": bit_flip_acc - results["base_accuracy"],
            "time_taken": end_time - start_time,
            "selected_experts": gate_outputs_trial["selected_experts"],
            "detailed_results": bit_flip_results
        }

        results["bit_flip_trials"].append(trial_data)
        results["selected_experts_trials"].append(gate_outputs_trial["selected_experts"])

        trial_results_path = os.path.join(output_dir, f"trial_{trial:03d}_results.json")
        save_detailed_results(trial_data, trial_results_path)

        if (trial + 1) % 10 == 0:
            avg_acc = np.mean([trial["accuracy"] for trial in results["bit_flip_trials"]])
            print(f"Interim average accuracy after {trial + 1} trials: {avg_acc:.4f}")

            save_detailed_results(results, os.path.join(output_dir, "all_results_interim.json"))

        progress_bar.update(1)

    if baseline_experts is not None and trial_experts_list:
        print("Creating expert selection comparison plots...")
        plot_experts_comparison(baseline_experts, trial_experts_list, output_dir, results)

    avg_accuracy = np.mean([trial["accuracy"] for trial in results["bit_flip_trials"]])
    accuracy_stddev = np.std([trial["accuracy"] for trial in results["bit_flip_trials"]])
    avg_time = np.mean([trial["time_taken"] for trial in results["bit_flip_trials"]])

    print("\n--- Final Results ---")
    print(f"Baseline accuracy: {results['base_accuracy']:.4f}")
    print(f"Average bit-flip accuracy: {avg_accuracy:.4f} (± {accuracy_stddev:.4f})")
    print(f"Average evaluation time: {avg_time:.2f} seconds")

    results["summary"] = {
        "avg_accuracy": float(avg_accuracy),
        "accuracy_stddev": float(accuracy_stddev),
        "avg_time": float(avg_time),
    }

    save_detailed_results(results, os.path.join(output_dir, "all_results_final.json"))

    summary_results = {
        "base_accuracy": results["base_accuracy"],
        "summary": results["summary"],
        "bit_flip_trials": [{k: v for k, v in trial.items() if k != 'detailed_results'}
                            for trial in results["bit_flip_trials"]]
    }
    save_detailed_results(summary_results, os.path.join(output_dir, "summary_results.json"))

    print("\nAll results saved to the 'moegategsm8k' directory")

if __name__ == "__main__":
    main() 