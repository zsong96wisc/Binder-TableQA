from typing import Dict, List, Union, Tuple, Optional
from generation.prompt import PromptBuilder
from llama import Llama, Dialog


class CodeGenerator:
    def __init__(self, args):
        self.args = args
        self.prompt_builder = PromptBuilder(args) if args else None


    def truncate_prompt(self, prompt: str, num_rows: int, end_token: str = '*/'):
        end_pos = prompt.rfind(end_token)
        assert end_pos != -1
        part1, part2 = prompt[:end_pos], prompt[end_pos:]
        part1_lines = part1.split('\n')[::-1]
        index = None
        for idx, line in enumerate(part1_lines):
            if '\t' in line:
                row_id = int(line.split('\t')[0])
                if row_id <= num_rows:
                    index = idx
                    break
        new_part1 = '\n'.join(part1_lines[index:][::-1])
        prompt = new_part1 + '\n' + part2
        return prompt
    
    def few_shot_prompt_from_file(self, file_path: str, n_shots: int):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        few_shot_prompts = []
        one_shot_prompt = ''
        last_line = None
        for line in lines:
            if line == '\n' and last_line == '\n':
                few_shot_prompts.append(one_shot_prompt.strip())
                one_shot_prompt = ''
            else:
                one_shot_prompt += line
            last_line = line
        few_shot_prompts = few_shot_prompts[:n_shots]
        return '\n'.join(few_shot_prompts)

    def generate_prompt(self, data_item: Dict, generate_type: Tuple):
        return self.prompt_builder.build_generate_prompt(**data_item, generate_type=generate_type)

    def generate(self, prompts: List[Tuple], verbose: bool = False):
        result_idx_to_eid = [p[0] for p in prompts for _ in range(self.args.sampling_n)]
        prompts = [p[1] for p in prompts]
        result = self.call_llama2(prompts)
        response_dict = self.parse_results(result, result_idx_to_eid)
        return response_dict

    def call_llama2(self, prompt: Union[str, List], max_tokens, temperature: float, top_p: float):
        generator = Llama.build(
            ckpt_dir="./llama-2-7b",
            tokenizer_path="/work/zhiwei/llama/tokenizer.model",
            max_seq_len=3800,
            max_batch_size=4,
        )
        result = None
        while result is None:
            try:
                choices = generator.text_completion(
                    [prompt] if isinstance(prompt, str) else prompt,
                    max_gen_len=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                result = {"choices": choices}
                return result
            except Exception as e:
                print(e, 'Retry.')

    def parse_results(self, result, result_idx_to_eid):
        response_dict = {}
        for idx, text in enumerate(result['choices']):
            try:
                logprob = 1 if 'logprobs' not in text else sum(text['logprobs']['token_logprobs'])
                eid = result_idx_to_eid[idx]
                eid_pairs = response_dict.setdefault(eid, [])
                eid_pairs.append((text, logprob))
            except Exception as e:
                pass
        return response_dict
