import guidance

from bazaar.lem_utils import LLaMa2

HF_AUTH = "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"
HF_CACHE_DIR = "/tmp/hf"
GUIDANCE_CACHE_DIR = "/tmp/gd"

llm = LLaMa2(
    hf_auth_token=HF_AUTH,
    hf_cache_directory=HF_CACHE_DIR,
    monitor_model=True,
    guidance_cache_directory=GUIDANCE_CACHE_DIR,
)

program_string = """
{{#role 'system'~}}
You are an intelligent AI assistant. You will be given a question. Your task is to answer it to the best of your ability. 
{{~/role}}

{{#role 'user'~}}
{{question}}
{{~/role}}

{{#assistant~}}
{{gen "answer" temperature=0.0 max_tokens=512}}
{{~/assistant}}
"""
# Run the program
program = guidance(program_string, llm=llm)  # noqa
program_output = program(question="What is the meaning of life?")
answer = program_output["answer"]

print(answer)
