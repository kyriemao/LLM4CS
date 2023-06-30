# eval_dense_retrieval.sh
# An example of evaluating RTR on cast20.
eval_field_name="predicted_rewrite"
work_dir="../results/cast20/RTR"    # set your the folder containing your `rewrites.jsonl`file

eval_file_path="$work_dir/rewrites.jsonl" \
index_path="" # set the pre-built index which contains all candidate passage emebddings. 
qrel_file_path="../datasets/cast20_qrel.tsv" # set the qrel file path
retrieval_output_path="$work_dir/ance/+q+r+sc" # set your expected output folder path

export CUDA_VISIBLE_DEVICES=0
python eval_dense_retrieval.py \
--eval_file_path=$eval_file_path \
--eval_field_name=$eval_field_name \
--qrel_file_path=$qrel_file_path \
--index_path=$index_path \
--retriever_path="" \ # set the pre-trained ANCE model path.
--use_gpu_in_faiss \
--n_gpu_for_faiss=1 \
--top_n=1000 \
--rel_threshold=2 \ # set 1 for cast19 and 2 for cast20.
--retrieval_output_path=$retrieval_output_path \
--include_query \
--include_response \ # enable `include_response` if you test RTR or RAR prompting.
--aggregation_method="sc" \ # you can set [`maxprob, mean, sc`]