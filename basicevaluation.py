import os
import random
import numpy as np
import torch
import time
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from torch.utils.data import DataLoader
from lm_eval import evaluator
import lm_eval
import re

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
    with torch.no_grad():
        tensor_bf16 = tensor.to(torch.bfloat16)
        bits = tensor_bf16.view(torch.int16)
        bits_value = bits.item()
        
        mask = (1 << bit_position[0]) | (1 << bit_position[1])
        bits = bits ^ mask
        
        flipped_tensor = bits.view(torch.bfloat16)
    return flipped_tensor

def get_input(dataset, tokenizer, id, device):
    question = dataset["question"][id]
    answer = dataset["answer"][id]
    #input = dataset["input_formatted"][id]
    prompt = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device, non_blocking=True)
    return answer, prompt, input_ids

def extract_final_answer(text):
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()

def extract_last_number(text):
    numbers = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
    if numbers:
        # 获取最后一个非空数字
        last_number = None
        for num in reversed(numbers):
            if num[0] or num[1]:  # 检查元组中的两个元素
                last_number = num[0] if num[0] else num[1]
                break
        if last_number:
            # 移除货币符号和逗号，转换为浮点数
            cleaned = last_number.replace('$', '').replace(',', '')
            try:
                return float(cleaned)
            except ValueError:
                return None
    return None

def generate(id, dataset, tokenizer, model, max_length=200):
    reference, prompt, input_ids = get_input(dataset, tokenizer, id, device)
    prompt_len = len(input_ids[0])

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=prompt_len + max_length,
            do_sample=False,
            num_beams=1,
            #temperature=0.9,
            #no_repeat_ngram_size=3,
            #num_return_sequences=1,
            #early_stopping=True,
            #pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    
    stop_tokens = ["Question:", "</s>", "<|im_end|>"]
    for stop_token in stop_tokens:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0]

    generated_text = generated_text.strip()
    reference_answer = extract_final_answer(reference)
    
    return generated_text, reference_answer, prompt

def is_answer_correct(prediction, reference):
    pred_number = extract_last_number(prediction)
    ref_number = float(reference)
    
    if pred_number is None or ref_number is None:
        return False
        
    return pred_number == ref_number

def read_qids_from_file(filename):
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file]

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    output_dir = "mfalcongsm8k"
    os.makedirs(output_dir, exist_ok=True)
    
    all_answers_file = os.path.join(output_dir, "all_answers.txt")
    different_answers_file = os.path.join(output_dir, "different_answers.txt")
    
    all_answers = open(all_answers_file, "w", encoding="utf-8")
    different_answers = open(different_answers_file, "w", encoding="utf-8")

    num_samples = 100


    dataset = datasets.load_dataset('tinyBenchmarks/tinyGSM8K', 'main')['test']
    max_id = len(dataset)
    sample_ids = random.sample(range(max_id), num_samples)

    #with open("gsm8k_indices.txt", "w") as f:
    #    for i in sample_ids:
    #        f.write(f"{i}\n")

    dataset = dataset.select(sample_ids)
    print(f"Dataset loaded with {len(dataset)} samples")

    print("Loading model and tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-7B-Instruct")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/Falcon3-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.resize_token_embeddings(len(tokenizer))
    print("Model and tokenizer loaded")
    for layer in model.children():
        print(layer)

    layer_weights = {
        'self_attn.v_proj': 2,
        'self_attn.k_proj': 2,
        'self_attn.q_proj': 6,
        'self_attn.o_proj': 6,
        'mlp.up_proj': 45,
        'mlp.gate_proj': 45,
        'mlp.down_proj': 45
    }

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
        prediction, reference, prompt = generate(idx, dataset, tokenizer, model, max_length=200)
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

    num_trials = 200
    print(f"Performing {num_trials} bit flip trials per sample...")

    bit_flip_progress = tqdm(total=len(dataset) * num_trials, desc="Bit flip trials")

    for trial in range(num_trials):
        layer_idx = random.randint(0, 27)

        selected_module = random.choices(layers, weights=weights)[0]

        target_layer = model.model.layers[layer_idx]

        module_path = selected_module.split('.')
        current_module = target_layer
        for path_part in module_path:
            current_module = getattr(current_module, path_part)
            
        weight_tensor = current_module.weight

        x = random.randint(0, weight_tensor.shape[0] - 1)
        y = random.randint(0, weight_tensor.shape[1] - 1)

        bit_position = random.sample(range(16), 2)

        original_weight_value = weight_tensor[x, y].clone()

        with torch.no_grad():
            weight_tensor[x, y] = perform_bit_flip(weight_tensor[x, y], bit_position)

        for sample_idx in range(len(dataset)):
            start_time = time.time()
            prediction, reference, prompt = generate(sample_idx, dataset, tokenizer, model, max_length=200)
            is_correct = is_answer_correct(prediction, reference)
            end_time = time.time()

            results["bit_flip_trials"].append({
                "sample_id": sample_idx,
                "layer_idx": layer_idx,
                "module": selected_module,
                "original_weight_value": original_weight_value,
                "flipped_weight_value": weight_tensor[x, y],
                "bit_position": bit_position,
                "reference": reference,
                "prediction": prediction,
                "is_correct": is_correct,
                "time_taken": end_time - start_time
            })
            
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
            
        with torch.no_grad():
            weight_tensor[x, y] = perform_bit_flip(weight_tensor[x, y], bit_position)
            
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
