import vllm
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95)
llm = vllm.LLM(model="meta-llama/Llama-2-70b-chat-hf")
output = llm.generate("Hello, my name is")
print(output)

