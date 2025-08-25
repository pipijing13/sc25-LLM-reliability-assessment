import os
import random
import numpy as np
import torch
import time
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import argparse
import re

# Set environment variables and random seed for reproducibility
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

# 2. nocot prompt

def get_input(dataset, tokenizer, id, device):
    question = dataset["question"][id]
    answer = dataset["answer"][id]
    prompt = f"Solve the following math problem, but output only the final numerical answer. Do not explain your reasoning process, do not show any calculation steps, just output a single number as the answer.\n Question: {question}\nAnswer:<|assistant|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device, non_blocking=True)
    return answer, prompt, input_ids


def extract_final_answer(text):
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def extract_last_number(text):
    numbers = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
    if numbers:
        last_number = None
        for num in reversed(numbers):
            if num[0] or num[1]:
                last_number = num[0] if num[0] else num[1]
                break
        if last_number:
            cleaned = last_number.replace('$', '').replace(',', '')
            try:
                return float(cleaned)
            except ValueError:
                return None
    return None

# 3. max_length=10, 4. stop_tokens扩展

def generate(id, dataset, tokenizer, model, max_length=10):
    reference, prompt, input_ids = get_input(dataset, tokenizer, id, device)
    prompt_len = len(input_ids[0])
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=prompt_len + max_length,
            do_sample=False,
            num_beams=1,
        )
    generated_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    stop_tokens = ["Question:", "</s>", "<|im_end|>", "<｜Assistant｜>", "<｜/Assistant｜>", "<|assistant|>"]
    for stop_token in stop_tokens:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0]
    generated_text = generated_text.strip()
    reference_answer = extract_final_answer(reference)
    return generated_text, reference_answer, prompt


def is_answer_correct(prediction, reference):
    pred_number = extract_last_number(prediction)
    try:
        ref_number = float(reference)
    except:
        return False
    if pred_number is None or ref_number is None:
        return False
    return pred_number == ref_number


def create_output_hook(module, bit_position, coordinates, token_position, fault_mode):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if not hasattr(hook, 'count'):
            hook.count = 0
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
    sequence_lengths = []
    min_tokens = float('inf')
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if output.shape[1] > 1:
            sequence_lengths.append(output.shape[1])
    first_layer = model.model.layers[0]
    module = first_layer.self_attn.v_proj
    hook = module.register_forward_hook(hook_fn)
    for idx in range(len(dataset)):
        _, _, input_ids = get_input(dataset, tokenizer, idx, device)
        prompt_len = len(input_ids[0])
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=prompt_len + 200,
                do_sample=False,
                num_beams=1,
            )
        generated_tokens = len(output[0][prompt_len:])
        min_tokens = min(min_tokens, generated_tokens)
    hook.remove()
    min_seq_len = min(sequence_lengths)
    print(f"Minimum sequence length observed: {min_seq_len}")
    print(f"Minimum generated tokens: {min_tokens}")
    return min_seq_len, min_tokens


def main():
    parser = argparse.ArgumentParser(description="GSM8K Fault Injection Experiment (NoCoT Mode)")
    parser.add_argument('--fault_mode', type=str, default='weight', choices=['weight', 'neuron', 'single'], help='Fault injection mode: weight (weight), neuron (neuron output double bit), single (neuron output single bit)')
    parser.add_argument('--model', type=str, default='falcon', choices=['falcon', 'qwen'], help='Model type: falcon or qwen')
    parser.add_argument('--num_trials', type=int, default=1000, help='Number of bit flip trials per sample')
    args = parser.parse_args()
    fault_mode = args.fault_mode
    model_type = args.model
    num_trials = args.num_trials

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    output_dir = f"nocotFI_{fault_mode}_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    all_answers_file = os.path.join(output_dir, "all_answers.txt")
    different_answers_file = os.path.join(output_dir, "different_answers.txt")
    all_answers = open(all_answers_file, "w", encoding="utf-8")
    different_answers = open(different_answers_file, "w", encoding="utf-8")

    num_samples = 100
    dataset = datasets.load_dataset('tinyBenchmarks/tinyGSM8K', 'main')['test']
    max_id = len(dataset)
    sample_ids = random.sample(range(max_id), num_samples)
    dataset = dataset.select(sample_ids)
    print(f"Dataset loaded with {len(dataset)} samples")

    print("Loading model and tokenizer...")
    global tokenizer
    if model_type == 'falcon':
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
        # Falcon structure
        layer_weights = {
            'self_attn.v_proj': 2,
            'self_attn.k_proj': 2,
            'self_attn.q_proj': 6,
            'self_attn.o_proj': 6,
            'mlp.up_proj': 45,
            'mlp.gate_proj': 45,
            'mlp.down_proj': 45
        }
        num_layers = 28
        def get_layer(idx):
            return model.model.layers[idx]
    elif model_type == 'qwen':
        import lm_eval
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
        # Qwen structure
        layer_weights = {
            'self_attn.v_proj': 1,
            'self_attn.k_proj': 1,
            'self_attn.q_proj': 7,
            'self_attn.o_proj': 7,
            'mlp.up_proj': 37,
            'mlp.gate_proj': 37,
            'mlp.down_proj': 37
        }
        num_layers = 28
        def get_layer(idx):
            return model.model.model.layers[idx]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print("Model and tokenizer loaded")
    for layer in range(num_layers):
        print(get_layer(layer))

    total_weight = sum(layer_weights.values())
    layers = list(layer_weights.keys())
    weights = [layer_weights[layer] / total_weight for layer in layers]

    results = {
        "baseline_answers": [],
        "bit_flip_trials": [],
    }
    baseline_answers = {}

    print("Generating baseline answers...")
    baseline_progress = tqdm(range(len(dataset)), desc="Baseline answers")
    total_baseline_correct = 0
    for idx in range(len(dataset)):
        prediction, reference, prompt = generate(idx, dataset, tokenizer, model, max_length=10)
        is_correct = is_answer_correct(prediction, reference)
        if is_correct:
            total_baseline_correct += 1
        results["baseline_answers"].append({
            "sample_id": idx,
            "reference": reference,
            "prediction": prediction,
            "is_correct": is_correct
        })
        all_answers.write(f"Sample ID: {idx} (Baseline)\n")
        all_answers.write(f"Question: {prompt}\n")
        all_answers.write(f"Reference: {reference}\n")
        all_answers.write(f"Prediction: {prediction}\n")
        all_answers.write(f"Correct: {is_correct}\n")
        all_answers.write("="*80 + "\n\n")
        baseline_answers[idx] = prediction
        baseline_progress.update(1)

    baseline_accuracy = total_baseline_correct / len(dataset)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")

    if fault_mode in ["neuron", "single"]:
        min_seq_len, min_tokens = record_output_dimensions(model, dataset, tokenizer, device)

    print(f"Performing {num_trials} bit flip trials per sample...")
    bit_flip_progress = tqdm(total=len(dataset) * num_trials, desc="Bit flip trials")

    for trial in range(num_trials):
        layer_idx = random.randint(0, num_layers - 1)
        selected_module = random.choices(layers, weights=weights)[0]
        target_layer = get_layer(layer_idx)
        module_path = selected_module.split('.')
        current_module = target_layer
        for path_part in module_path:
            current_module = getattr(current_module, path_part)
        weight_tensor = current_module.weight

        if fault_mode == 'weight':
            x = random.randint(0, weight_tensor.shape[0] - 1)
            y = random.randint(0, weight_tensor.shape[1] - 1)
            bit_position = random.sample(range(16), 2)
            original_weight_value = weight_tensor[x, y].clone()
            with torch.no_grad():
                weight_tensor[x, y] = perform_bit_flip_weight(weight_tensor[x, y], bit_position)
        else:
            if fault_mode == 'neuron':
                bit_position = random.sample(range(16), 2)
            elif fault_mode == 'single':
                bit_position = random.randint(0, 15)
            token_position = random.randint(0, min_tokens - 1)
            if token_position == 0:
                x = random.randint(0, min_seq_len - 1)
            else:
                x = 0
            y = random.randint(0, weight_tensor.shape[0] - 1)
            hook = create_output_hook(current_module, bit_position, (x, y), token_position, fault_mode)
            hook_handle = current_module.register_forward_hook(hook)

        for sample_idx in range(len(dataset)):
            if fault_mode in ["neuron", "single"] and hasattr(hook, 'count'):
                delattr(hook, 'count')
            start_time = time.time()
            prediction, reference, prompt = generate(sample_idx, dataset, tokenizer, model, max_length=10)
            is_correct = is_answer_correct(prediction, reference)
            end_time = time.time()
            trial_result = {
                "sample_id": sample_idx,
                "layer_idx": layer_idx,
                "module": selected_module,
                "bit_position": bit_position,
                "reference": reference,
                "prediction": prediction,
                "is_correct": is_correct,
                "time_taken": end_time - start_time
            }
            if fault_mode == 'weight':
                trial_result["original_weight_value"] = original_weight_value
                trial_result["flipped_weight_value"] = weight_tensor[x, y].clone()
            results["bit_flip_trials"].append(trial_result)
            all_answers.write(f"Sample ID: {sample_idx} (Bit Flip)\n")
            all_answers.write(f"Layer: {layer_idx}, Module: {selected_module}, Bit: {bit_position}\n")
            all_answers.write(f"Question: {prompt}\n")
            all_answers.write(f"Reference: {reference}\n")
            all_answers.write(f"Prediction: {prediction}\n")
            all_answers.write(f"Correct: {is_correct}\n")
            all_answers.write("="*80 + "\n\n")
            if prediction != baseline_answers[sample_idx]:
                different_answers.write(f"Sample ID: {sample_idx}\n")
                different_answers.write(f"Layer: {layer_idx}, Module: {selected_module}, Bit: {bit_position}\n")
                different_answers.write(f"Question: {prompt}\n")
                different_answers.write(f"Reference: {reference}\n")
                different_answers.write(f"Baseline: {baseline_answers[sample_idx]}\n")
                different_answers.write(f"Bit-flip: {prediction}\n")
                different_answers.write(f"Correct: {is_correct}\n")
                different_answers.write("="*80 + "\n\n")
            del prediction
            del reference
            torch.cuda.empty_cache()
            bit_flip_progress.update(1)
            if bit_flip_progress.n % 100 == 0:
                correct_trials = [t for t in results["bit_flip_trials"] if t["is_correct"]]
                if correct_trials:
                    accuracy = len(correct_trials) / len(results["bit_flip_trials"])
                    print(f"\nInterim accuracy after {bit_flip_progress.n} trials: {accuracy:.4f}")

        if fault_mode == 'weight':
            with torch.no_grad():
                weight_tensor[x, y] = perform_bit_flip_weight(weight_tensor[x, y], bit_position)
        else:
            hook_handle.remove()
        torch.cuda.empty_cache()

    all_answers.close()
    different_answers.close()

    correct_trials = [t for t in results["bit_flip_trials"] if t["is_correct"]]
    bit_flip_accuracy = len(correct_trials) / len(results["bit_flip_trials"])
    avg_time = sum(t["time_taken"] for t in results["bit_flip_trials"]) / len(results["bit_flip_trials"])
    print("\n--- Final Results ---")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Bit-flip accuracy: {bit_flip_accuracy:.4f}")
    print(f"Average evaluation time: {avg_time:.2f} seconds")
    print(f"All answers saved to: {all_answers_file}")
    print(f"Different answers saved to: {different_answers_file}")

    module_impacts = {}
    for module in layer_weights.keys():
        module_trials = [t for t in results["bit_flip_trials"] if t["module"] == module]
        if module_trials:
            module_accuracy = len([t for t in module_trials if t["is_correct"]]) / len(module_trials)
            module_impacts[module] = {
                "accuracy": module_accuracy
            }
    print("\n--- Impact by Module Type ---")
    for module, scores in module_impacts.items():
        print(f"{module}:")
        print(f"  Accuracy: {scores['accuracy']:.4f} (Δ from baseline: {scores['accuracy'] - baseline_accuracy:.4f})")


if __name__ == "__main__":
    main() 