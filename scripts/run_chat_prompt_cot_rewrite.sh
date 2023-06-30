python chat_prompt_cot_rewrite.py \
--open_ai_key_id=0 \
--test_file_path="./datasets/cast21_test.json" \
--demo_file_path="./demonstrations.json" \
--qrel_file_path="./datasets/cast21_qrel.tsv" \
--work_dir="./results/new/cast21/COT_REW" \
--n_generation=5 \