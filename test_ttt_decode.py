import time
from transformers.models.ttt.modeling_ttt import TttForCausalLM
from transformers import AutoTokenizer, GenerationConfig

model_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts_release/LLAMA-125M/06_01_Tok_llama2_D_2.5B_ctx_2048_BS_256_M1_Dual_out_norm_ln_learnable_row_transpose_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_1/hf_4800"
# model_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts/LLAMA-125M/06_01_Tok_llama2_D_2.5B_ctx_2048_BS_256_c1d_k4_M1MixerLinear_Dual_out_norm_ln_share_qk_qk4_learnable_row_transpose_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_1/hf_4800"
# model_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts_release/LLAMA-125M/06_01_Tok_llama2_D_2.5B_ctx_2048_BS_256_M2_Dual_out_norm_ln_learnable_row_transpose_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_0.01_480_to_0.1/hf_4800"
# model_path = "/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts_release/LLAMA-125M/06_01_Tok_llama2_D_2.5B_ctx_2048_BS_256_c1d_k4_M2MixerLinear_Dual_out_norm_ln_share_qk_qk4_learnable_row_transpose_postln_res_chunk_rotary_lr_3e-3_ilr_sigmoid_0.01_480_to_0.1/hf_4800"

if __name__ == '__main__':
    print('loading', model_path)
    # config = TttConfig.from_dict(TTT_STANDARD_CONFIGS['debug'])
    # model = TttForCausalLM(config)
    model = TttForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to('cuda')
    
    if tokenizer.pad_token:
        print('pad token:', tokenizer.pad_token)
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        print('pad token to unk token', tokenizer.pad_token_id)
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print('pad token to eos token', tokenizer.pad_token_id)
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = 'right'
    print('pad_token_id:', tokenizer.pad_token_id)
    # prompt = ["Hey, are you conscious? Can you talk to me?", "I am Jerry."]
    prompt = "Hey, are you conscious? Can you talk to me?"
    input_length = 12
    test_length = 100
    # test_length = 13
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=input_length, truncation=True).to('cuda')
    print('inputs:', inputs.input_ids.shape)
    generation_config=GenerationConfig(pad_token_id=0)

    # Generate using cache
    start_time = time.time()
    cache_generate_ids = model.generate(**inputs, max_length=test_length, min_length=test_length, do_sample=False, use_cache=True, generation_config=generation_config)
    print('Time taken to generate using cache:', time.time() - start_time)

    # Generate without using cache
    start_time = time.time()
    no_cache_generate_ids = model.generate(**inputs, max_length=test_length, min_length=test_length, do_sample=False, use_cache=False, generation_config=generation_config)
    print('Time taken to generate without cache:', time.time() - start_time)


    print('no_cache_generate_ids:', no_cache_generate_ids.shape)
    print(tokenizer.batch_decode(no_cache_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    print('cache_generate_ids:', cache_generate_ids.shape)
    print(tokenizer.batch_decode(cache_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    print(cache_generate_ids - no_cache_generate_ids)
    print(cache_generate_ids[0, inputs.input_ids.shape[1]:] - no_cache_generate_ids[0, inputs.input_ids.shape[1]:])
