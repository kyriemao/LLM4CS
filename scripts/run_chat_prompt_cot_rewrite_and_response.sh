python chat_prompt_cot_rewrite_and_response.py \
--open_ai_key_id=1 \
--qrel_file_path="./datasets/cast21_qrel.tsv" \
--test_file_path="./datasets/cast21_test.json" \
--demo_file_path="./demonstrations.json" \
--work_dir="./results/new/cast21/COT_RAR" \
--n_generation=5 \