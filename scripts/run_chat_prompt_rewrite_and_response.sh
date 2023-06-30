python chat_prompt_rewrite_and_response.py \
--open_ai_key_id=0 \
--qrel_file_path="./datasets/cast20_qrel.tsv" \
--test_file_path="./datasets/cast20_test.json" \
--demo_file_path="./demonstrations.json" \
--work_dir="./results/new/cast20/RAR" \
--n_generation=5 \