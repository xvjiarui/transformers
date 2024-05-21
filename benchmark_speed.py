'''Benchmarking Prefilling and Decoding

=== prefill ===
# Throughput ~= 156488.003
python benchmark_speed.py --logdir ./exp/prefill_ttt_125m \
                            --mode prefill \
                            --model-name ttt-125m \
                            --inner_net m1 \
                            --batch 64 \
                            --promptlen 512 \
                            --genlen 0

# Throughput ~= 415172.625
python benchmark_speed.py --logdir ./exp/prefill_llama_125m \
                            --mode prefill \
                            --model-name llama-125m \
                            --batch 64 \
                            --promptlen 512 \
                            --genlen 0

# Throughput ~= 354626.443 (takes 5min to compile)
python benchmark_speed.py --logdir ./exp/prefill_ttt_125m \
                            --mode prefill \
                            --model-name ttt-125m \
                            --inner_net m1 \
                            --batch 64 \
                            --promptlen 512 \
                            --genlen 0 \
                            --use_compile

# Throughput ~= 34264.801
python benchmark_speed.py --logdir ./exp/prefill_ttt_1b \
                            --mode prefill \
                            --model-name ttt-1b \
                            --inner_net m1 \
                            --batch 64 \
                            --promptlen 512 \
                            --genlen 0


# Throughput ~= 33476.099
python benchmark_speed.py --logdir ./exp/prefill_ttt_1b \
                            --mode prefill \
                            --model-name ttt-1b \
                            --inner_net m1 \
                            --batch 64 \
                            --promptlen 1024 \
                            --genlen 0

# Throughput ~= 33166.213
python benchmark_speed.py --logdir ./exp/prefill_ttt_1b \
                            --mode prefill \
                            --model-name ttt-1b \
                            --inner_net m1 \
                            --batch 64 \
                            --promptlen 2048 \
                            --genlen 0

# Throughput ~= 53585.024 (takes 10min to compile)
python benchmark_spped.py --logdir ./exp/prefill_ttt_1b \
                            --mode prefill \
                            --model-name ttt-1b \
                            --inner_net m1 \
                            --batch 64 \
                            --promptlen 512 \
                            --genlen 0 \
                            --use_compile

# Throughput ~= 46178.843  (takes 10min to compile)
python benchmark_prefill.py --logdir ./exp/prefill_ttt_1b \
                            --mode prefill \
                            --model-name ttt-1b \
                            --inner_net m1 \
                            --batch 64 \
                            --promptlen 1024 \
                            --genlen 0 \
                            --use_compile

# Throughput ~= 75578.140 (very fast compare to TTT)
python benchmark_speed.py --logdir ./exp/prefill_llama_1b \
                            --mode prefill \
                            --model-name llama-1b \
                            --batch 64 \
                            --promptlen 512 \
                            --genlen 0 \
                            --use_compile

=== decode ===

# Throughput ~= 3049.353
python benchmark_speed.py --logdir ./exp/decode_ttt_125m \
                            --mode decode \
                            --model-name ttt-125m \
                            --inner_net m1 \
                            --batch 64 \
                            --promptlen 1 \
                            --genlen 512
'''

import argparse
import os
import os.path as osp
import time
import json
import logging

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PretrainedConfig
from transformers import LlamaForCausalLM, LlamaConfig

from transformers.models.ttt.configuration_ttt import TTT_STANDARD_CONFIGS, TttConfig  # 125m and 1b config

from transformers.models.ttt.modeling_ttt import TttForCausalLM

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--logdir", type=str, default="./exp/clean")
parser.add_argument("--model-name", type=str, default="ttt-1b")
# state-spaces/mamba-130m | meta-llama/Llama-2-7b | state-spaces/mamba-1.4b | ttt-125m | ttt-1b | ttt-profile
parser.add_argument("--mode", type=str, default="prefill", choices=["prefill", "decode"])
parser.add_argument("--promptlen", type=int, default=1)
parser.add_argument("--genlen", type=int, default=128)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--attn_impl", type=str, default='flash_attention_2', choices=['eager', 'flash_attention_2'])
parser.add_argument("--inner_net", type=str, default='m1', choices=['m1','m2'])
parser.add_argument("--use_compile", action='store_true')
parser.add_argument("--profile", action='store_true')  # @xinhao: pytorch profiler, different from nsys in micro-benchmark
args = parser.parse_args()

def print_args(args):
    s = "Arguments: \n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

os.makedirs(args.logdir, mode=0o777, exist_ok=True)

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to the lowest level to catch all messages
file_handler = logging.FileHandler(osp.join(args.logdir, "logging.log"), mode='a')
console_handler = logging.StreamHandler()
# Create a formatter and set it on handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Test messages
logger.info(f"\n============== Model Name: {args.model_name} ==============")
config_str = print_args(args)
logger.info(config_str)

torch.random.manual_seed(0)  # @xinhao: make sure model init is fixed

repeats = 10
device = "cuda"
dtype = torch.float16  # @xinhao: follow mamba benchmark
logger.info("dtype: " + str(dtype))

logger.info(f"Loading model {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if args.model_name.startswith("ttt"):
    ttt_size = args.model_name.split('-')[-1]
    ttt_config = TttConfig(**TTT_STANDARD_CONFIGS[ttt_size])
    ttt_config.inner_net = args.inner_net
    model = TttForCausalLM(ttt_config).to(device=device, dtype=dtype)
elif args.model_name.startswith("llama"):
    llama_size = args.model_name.split('-')[-1]
    config = LlamaConfig(**TTT_STANDARD_CONFIGS[llama_size])
    config._attn_implementation = args.attn_impl  # @xinhao: llama config use `_attn_implementation` to select attn
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype)

logger.info(model)
model.eval()
logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device=device)
attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
max_length = input_ids.shape[1] + args.genlen

if args.use_compile:
    model = torch.compile(model)  # @xinhao: can compile the whole Transformer for decode, though doesn't help
if args.mode == 'decode':
    fn = lambda i: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        min_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        use_cache=True,
    )
elif args.mode == 'prefill':
    kwargs = {'input_ids': input_ids}
    @torch.inference_mode()
    def fn(*args):
        model(**kwargs, use_cache=False)
        return
else:
    raise NotImplementedError(f"Invalid Mode {args.mode}!")

out = fn(0)  # capture graph if cg=True, will not be timed
if args.mode == 'decode':
    logger.info(f"Decode succeeds. output.sequences.shape: {out.sequences.shape}")
    out_len = len(out.sequences[0])
else:
    logger.info("Prefill succeeds.")
    out_len = len(input_ids[0])
in_len = len(input_ids[0])
if args.mode == 'decode':
    print(tokenizer.batch_decode(out.sequences[0].tolist()[:5]))
    print('=================')
del out

if args.profile:

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True, use_cuda=True,
                 with_flops=True, with_stack=True, with_modules=True
        ) as prof:
           fn(0)
    prof.export_chrome_trace(osp.join(args.logdir, f"{args.mode}_trace.json"))
    prof.export_stacks(osp.join(args.logdir, f"{args.mode}_cuda_flamedata.txt"), "self_cuda_time_total")
    prof.export_stacks(osp.join(args.logdir, f"{args.mode}_cpu_flamedata.txt"), "self_cpu_time_total")
    torch.cuda.synchronize()

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Prompt length: {in_len}, generation length: {out_len - in_len}")
    logger.info(f"SUCCESS: RECORDED TRACE TO {args.logdir}/{args.mode}_trace.json")
    logger.info(f"SUCCESS: RECORDED FLAME DATA TO {args.logdir}/{args.mode}_[cuda,cpu]_flamedata.txt")
    logger.info("==================================")

else:

    torch.cuda.synchronize()
    start = time.time()
    for i in range(repeats):
        fn(i)
    torch.cuda.synchronize()
    avg_time = (time.time() - start) / repeats

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Prompt length: {in_len}, generation length: {out_len - in_len}")
    logger.info(f"{args.model_name} prompt processing + decoding time: {avg_time * 1000:.0f}ms")
    logger.info(f"Throughput (total tok = prefill + decode): {args.batch * out_len / avg_time:.3f} tokens / s")
    logger.info(f"Throughput (total tok = decode): {args.batch * (out_len - in_len) / avg_time:.3f} tokens / s")
    logger.info("==================================")