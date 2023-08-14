import os
import guidance
import transformers
import bitsandbytes
from torch import cuda, bfloat16
from bazaar.schema import Quote
from bazaar.lem_utils import clean_program_string



class LLaMa2(guidance.llms.Transformers): 
    llm_name: str = "llama2"
    default_system_prompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""
    
    def __init__(self):
        hf_auth = "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"
        # model_id = 'meta-llama/Llama-2-70b-chat-hf'
        model_id = '/Tmp/slurm.3479725.0/hf_home/hub/models--meta-llama--Llama-2-70b-chat-hf/snapshots/36d9a7388cc80e5f4b3e9701ca2f250d21a96c30/'
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        tokenizer_name = "meta-llama/Llama-2-70b-chat-hf"
        
        # Build model
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )
        model.eval()
        print(f"Model loaded on {device}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=hf_auth)
        super().__init__(model=model, tokenizer=tokenizer, chat_mode=True)

    @staticmethod
    def role_start(role):
        if role == 'user':
            return 'USER: '
        elif role == 'assistant':
            return 'ASSISTANT: '
        else:
            return ''
    
    @staticmethod
    def role_end(role):
        if role == 'user':
            return ''
        elif role == 'assistant':
            return '</s>'
        else:
            return ''


guidance.llm = LLaMa2()

# Prompt
statement="Dictionary based approaches adapt the bag of words model commonly used in signal processing, computer vision and audio processing for time series classification (TSC).     A comparison of TSC algorithms, commonly known as the bake off, formed a taxonomy of approaches based on representations of discriminatory features, with dictionary approaches being one of these.     From the bake off the bag of Symbolic-Fourier Approximation symbols (BOSS)\u00a0[schafer2015boss] ensemble was found to be the most accurate dictionary classifier by a significant amount.     BOSS was found to be the third most accurate algorithm out of the 20 compared. This highlights the utility of dictionary methods for TSC.          This performance lead to BOSS being incorporated into the hierarchical vote collective of transformation-based ensembles (HIVE-COTE)\u00a0[lines2018time], a heterogeneous ensemble encompassing multiple representations.     The inclusion of BOSS and the subsequent significant improvement in accuracy places HIVE-COTE in the state of the art for TSC among three other algorithms proposed more recently.     These are the time series combination of heterogeneous and integrated embeddings forest (TS-CHIEF)\u00a0[shifaz2020ts], which also a hybrid of multiple representations, the random convolutional kernel transform (ROCKET)\u00a0[dempster2019rocket], and the deep learning approach InceptionTime\u00a0[fawaz2019inceptiontime].          Since the bake off a number of dictionary algorithms have been published, focusing on improving accuracy\u00a0[schafer2017fast,large2019time], prediction time efficiency\u00a0[schafer2017fast], train time and memory efficiency\u00a0[middlehurst2019scalable].     These algorithms are mostly extensions of BOSS, making alterations to different parts of the original algorithm.     Word extraction for time series classification (WEASEL)\u00a0[schafer2017fast] abandons the ensemble structure in favour of feature selection and changes the method of word discretisation.     Spatial BOSS (S-BOSS)\u00a0[large2019time] introduces temporal information and additional features using spatial pyramids.     Contractable BOSS (cBOSS)\u00a0[middlehurst2019scalable] changes the method used by BOSS to form its ensemble to improve efficiency and allow for a number of usability improvements.          Each of these methods constitutes an improvement to the dictionary representation from BOSS. Our contribution is to combine design features of these four classifiers (BOSS, WEASEL, S-BOSS and cBOSS) to make a new algorithm, the Temporal Dictionary Ensemble (TDE). Like BOSS, TDE is a homogeneous ensemble of nearest neighbour classifiers that use distance between histograms of word counts and injects diversity through parameter variation. TDE takes the ensemble structure from cBOSS, which is more robust and scaleable. The use of spatial pyramids is adapted from S-BOSS. WEASEL uses bi-grams and an alternative method of finding word breakpoints. This too is employed by TDE.           We found the simplest way of combining these components did not result in significant improvement. We speculate that the massive increase in the parameter space made the randomised diversity mechanism result in too many poor learners in the ensemble. We propose a novel mechanism of base classifier model selection based on an adaptive form of Gaussian process (GP) modelling of the parameter space. Through extensive evaluation with the UCR time series classification repository\u00a0[dau2019ucr], we show that  TDE is significantly more accurate than WEASEL and S-BOSS while retaining the usability and scalability of cBOSS. We further show that if TDE replaces BOSS in HIVE-COTE, the resulting classifier is significantly more accurate than HIVE-COTE with BOSS and all three competing state of the art classifiers.          The rest of this paper is structured as follows.      Section\u00a0<ref> provides background information for the four dictionary based algorithms relevant to TDE.      Section\u00a0<ref> describes the TDE algorithm, including the GP based parameter search. Section\u00a0<ref> presents the performance evaluation of TDE.      Conclusions are drawn in Section\u00a0<ref> and future work is discussed."

program_string = """
{{#system~}}
Socrates and Plato sit under a tree, discussing the nature of truth and knowledge. They have a scroll in front of them containing scientific texts. Socrates believes in extracting questions and answers that are factual and based on the content of the text. Plato, on the other hand, emphasizes that these answers must be objective assertions that describe reality and are supported by evidence.

Socrates: "Knowledge, my dear Plato, must be empirical and verifiable. Our task is to extract questions and answers from this scroll that adhere to this principle."

Plato: "Agreed, Socrates. But each answer must be comprehensive, providing context and depth. They should be reminiscent of the great archives, like an excerpt from our Athenian repositories."
{{~/system}}

{{#user~}}
This scientific text states: {{statement}}.

Now, my dear philosophers, deliberate upon the factual statements and pose questions based on the content. And then, provide corresponding factual answers.
At the end of the argument, they must arrive at a verdict. This verdict must be printed as: 

VERDICT:
question: \<answer\>
answer: \<answer\>

{{~/user}}

{{#assistant~}}
{{gen "answer" temperature=0.0 max_tokens=512}}
{{~/assistant}}
"""
program_string = clean_program_string(program_string)
program = guidance(program_string)
program_output = program(statement=statement)
print(program_output)


