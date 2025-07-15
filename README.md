# sc25-LLM-reliability-assessment

**PyTorch 2.5**, other required packages are shown in requirements.txt

## Command example:
```bash
python multichoiceFI.py –fault_mode neuron –model qwen –num_trials 1000
```

### `–fault_mode` (default: weight)
Specifies the fault injection mode
- **weight**: Inject faults into model weights
- **neuron**: Inject double-bit faults into neuron outputs
- **single**: Inject single-bit faults into neuron outputs

### `–generation_mode` (default: greedy)
Text generation strategy
- **beam**: Use beam search for generation
- **greedy**: Use greedy decoding for generation

### `–model`
Target model for fault injection
- **alma**: ALMA-7B model
- **qwen**: Qwen-7B model
- **llama2**: LLaMA2-7B model
- **llama3**: Llama-3.1-8B model
- **falcon**: Falcon3-7B model
- **summarizer**: Llama-3.1-8B-Summarizer model

### `–task`
Target dataset for fault injection
- **mmlu**: MMLU
- **arc**: AI2_ARC
- **hella**: HellaSwag
- **wino**: WinoGrande
- **truth**: TruthfulQA

### `–num_trials` (default: 500/1000)
Number of bit flip experiments to perform per input sample
