python chat_prompt_rewrite_then_response.py \
--open_ai_key_id=0 \
--test_file_path="./datasets/cast19_test.json" \
--qrel_file_path="./datasets/cast19_qrel.tsv" \
--work_dir="./results/new/cast19/RTR" \
--demo_file_path="./demonstrations.json" \
--rewrite_file_path="./results/new/cast19/REW/rewrites.jsonl" \
--n_generation=5 \