import re
import os
from collections import defaultdict, Counter

def extract_last_number(text):
    """从文本中提取最后一个数字。"""
    if text is None:
        return None
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

def is_answer_correct(prediction, reference):
    """检查答案是否正确（基于数值比较）。"""
    pred_number = extract_last_number(prediction)
    ref_number = extract_last_number(reference)
    
    if pred_number is None or ref_number is None:
        return False
        
    return abs(pred_number - ref_number) < 1e-6  

def has_infinite_loop(text):
    """检测输出中是否有无限循环的迹象（重复特殊字符）。"""
    if text is None:
        return True
    
    special_chars = r'[^\w\s]'
    special_char_count = len(re.findall(special_chars, text))
    total_length = len(text) if text else 0
    
    if total_length > 0 and special_char_count / total_length > 0.6:
        return True
    
    pattern_len = min(10, len(text) // 4) if text else 0
    if pattern_len > 2:
        for i in range(2, pattern_len + 1):
            pattern = text[:i]
            repeats = text.count(pattern)
            if repeats > 10 and repeats * len(pattern) > len(text) * 0.1:
                return True
                
    return False

def analyze_different_answers(file_path):
    """分析different_answers.txt文件中的结果。"""
    results = {
        'text_changed_result_same': 0,    
        'result_changed': 0,            
        'infinite_loop': 0,             
        'baseline_correct': 0,              
        'bitflip_correct': 0,              
        'baseline_correct_bitflip_wrong': 0, 
        'baseline_wrong_bitflip_correct': 0, 
        'both_wrong_but_different': 0,       
        'total_samples': 0                   
    }
    
    modules_affected = defaultdict(int)  
    layers_affected = defaultdict(int)    
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    samples = content.split('=' * 80)
    
    for sample in samples:
        if not sample.strip():
            continue
            
        results['total_samples'] += 1
        
        sample_id_match = re.search(r'Sample ID: (\d+)', sample)
        layer_module_match = re.search(r'Layer: (\d+), Module: ([^,]+)', sample)
        reference_match = re.search(r'Reference: ([\s\S]*?)(?=\nBaseline:|$)', sample)
        baseline_match = re.search(r'Baseline: ([\s\S]*?)(?=\nBit-flip:|$)', sample)
        bitflip_match = re.search(r'Bit-flip: ([\s\S]*?)(?=\nCorrect:|$)', sample)
        correct_match = re.search(r'Correct: (True|False)', sample)
        
        if not (sample_id_match and layer_module_match and reference_match and 
                baseline_match and bitflip_match):
            continue
            
        sample_id = sample_id_match.group(1)
        layer = layer_module_match.group(1)
        module = layer_module_match.group(2)
        reference_text = reference_match.group(1)
        baseline_text = baseline_match.group(1)
        bitflip_text = bitflip_match.group(1)
        is_correct = correct_match.group(1) == 'True' if correct_match else False
        
        reference_number = extract_last_number(reference_text)
        baseline_number = extract_last_number(baseline_text)
        bitflip_number = extract_last_number(bitflip_text)
        
        modules_affected[module] += 1
        layers_affected[layer] += 1
        
        baseline_correct = is_answer_correct(baseline_text, reference_text)
        bitflip_correct = is_answer_correct(bitflip_text, reference_text)
        
        if baseline_correct:
            results['baseline_correct'] += 1
        
        if bitflip_correct:
            results['bitflip_correct'] += 1
        
        if has_infinite_loop(bitflip_text):
            results['infinite_loop'] += 1

        elif baseline_number == bitflip_number:
            results['text_changed_result_same'] += 1
        else:
            results['result_changed'] += 1
        
        if baseline_correct and not bitflip_correct:
            results['baseline_correct_bitflip_wrong'] += 1
        elif not baseline_correct and bitflip_correct:
            results['baseline_wrong_bitflip_correct'] += 1
        elif not baseline_correct and not bitflip_correct and baseline_number != bitflip_number:
            results['both_wrong_but_different'] += 1
    
    return results, dict(modules_affected), dict(layers_affected)

def main():
    file_path = 'wmt16weights/different_translations.txt'
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return
    
    results, modules_affected, layers_affected = analyze_different_answers(file_path)
    
    print("=" * 50)
    print("位翻转实验结果分析")
    print("=" * 50)
    
    print(f"\n总样本数: {results['total_samples']}")
    
    print("\n1. 输出变化分类：")
    print(f"  - 文本变化但结果相同: {results['text_changed_result_same']} ({results['text_changed_result_same']/results['total_samples']*100:.2f}%)")
    print(f"  - 结果变化: {results['result_changed']} ({results['result_changed']/results['total_samples']*100:.2f}%)")
    print(f"  - 无限循环/特殊字符: {results['infinite_loop']} ({results['infinite_loop']/results['total_samples']*100:.2f}%)")
    
    print("\n2. 正确性分析：")
    print(f"  - 基准答案正确: {results['baseline_correct']} ({results['baseline_correct']/results['total_samples']*100:.2f}%)")
    print(f"  - 位翻转后答案正确: {results['bitflip_correct']} ({results['bitflip_correct']/results['total_samples']*100:.2f}%)")
    print(f"  - 基准正确但位翻转错误: {results['baseline_correct_bitflip_wrong']} ({results['baseline_correct_bitflip_wrong']/results['total_samples']*100:.2f}%)")
    print(f"  - 基准错误但位翻转正确: {results['baseline_wrong_bitflip_correct']} ({results['baseline_wrong_bitflip_correct']/results['total_samples']*100:.2f}%)")
    print(f"  - 两者都错但答案不同: {results['both_wrong_but_different']} ({results['both_wrong_but_different']/results['total_samples']*100:.2f}%)")
    
    print("\n3. 受影响最多的模块：")
    sorted_modules = sorted(modules_affected.items(), key=lambda x: x[1], reverse=True)
    for module, count in sorted_modules[:5]:
        print(f"  - {module}: {count} ({count/results['total_samples']*100:.2f}%)")
    
    print("\n4. 受影响最多的层：")
    sorted_layers = sorted(layers_affected.items(), key=lambda x: x[1], reverse=True)
    for layer, count in sorted_layers[:5]:
        print(f"  - 层 {layer}: {count} ({count/results['total_samples']*100:.2f}%)")

if __name__ == "__main__":
    main() 
