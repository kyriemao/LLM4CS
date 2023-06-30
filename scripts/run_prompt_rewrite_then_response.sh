python prompt_rewrite_then_response.py \
--open_ai_key_id=0 \
--demonstration_file_path="./demonstrations.json" \
--test_file_path="./datasets/cast20_test.json" \
--work_dir="./results/cast20/RTR" \
--model_name="code-davinci-002" \
--rewrite_file_path="./results/cast20/REW/rewrites.jsonl" \
--n_completion=5 \


# python prompt_rewrite_then_response.py \
# --open_ai_key_id=0 \
# --demonstration_file_path="./demonstrations.json" \
# --test_file_path="./datasets/cast20_test.json" \
# --work_dir="./results/cast20/COT-RTR" \
# --model_name="code-davinci-002" \
# --rewrite_file_path="./results/cast20/COT-REW/rewrites.jsonl" \
# --n_completion=5 \
