import os
import json
import pytrec_eval
import numpy as np
from pprint import pprint
from IPython import embed

def trec_eval(run_trec_file, qrel_trec_file, retrieval_output_path, rel_threshold):
    # process run trec file
    with open(run_trec_file, 'r' )as f:
        run_data = f.readlines()
    runs = {}
    for line in run_data:
        line = line.split(" ")
        sample_id = line[0]
        doc_id = line[2]
        score = float(line[4])
        if sample_id not in runs:
            runs[sample_id] = {}
        runs[sample_id][doc_id] = score

    # process qrel trec file
    with open(qrel_trec_file, 'r') as f:
        qrel_data = f.readlines()
    qrels = {}
    qrels_ndcg = {}
    for line in qrel_data:
        line = line.strip().split("\t")
        query = line[0]
        doc_id = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][doc_id] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][doc_id] = rel
 

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list),
            "NDCG@3": np.average(ndcg_3_list), 
        }

    
    print("---------------------Evaluation results:---------------------")    
    pprint(res)
    if retrieval_output_path:
        with open(os.path.join(retrieval_output_path, "metric.res"), "w") as f:
            f.write(json.dumps(res, indent=4))

    return res



if __name__ == "__main__":
    pass