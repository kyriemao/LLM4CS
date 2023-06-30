# from IPython import embed
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import pickle
import sys
sys.path.append('..')
sys.path.append('.')

import json
import time
import copy
import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from pprint import pprint

from trec_eval import trec_eval
from dense_retrievers import load_dense_retriever
from utils import set_seed, get_has_qrel_label_sample_ids


def build_faiss_index(args):
    logger.info("Building Faiss Index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.n_gpu_for_faiss
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(768)  
    index = None
    if args.use_gpu_in_faiss:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    return index

def get_embeddings(args):
    def query_encoding_collate_fn(batch):
        bt_sample_ids, bt_src_query = list(zip(*batch)) # unzip
        bt_src_query_encoding = tokenizer(bt_src_query, 
                                          padding="longest", 
                                          max_length=args.max_query_length, 
                                          truncation=True, 
                                          return_tensors="pt")
        
        bt_q_input_ids, bt_q_attention_mask = bt_src_query_encoding.input_ids, bt_src_query_encoding.attention_mask
       
        return {"bt_sample_ids": bt_sample_ids, 
                "bt_input_ids":bt_q_input_ids, 
                "bt_attention_mask":bt_q_attention_mask}
    
    def response_encoding_collate_fn(batch):
        bt_sample_ids, bt_src_doc = list(zip(*batch)) # unzip
        bt_src_doc_encoding = tokenizer(bt_src_doc, 
                                          padding="longest", 
                                          max_length=512, 
                                          truncation=True, 
                                          return_tensors="pt")
        bt_d_input_ids, bt_d_attention_mask = bt_src_doc_encoding.input_ids, bt_src_doc_encoding.attention_mask
        return {"bt_sample_ids": bt_sample_ids, 
                "bt_input_ids":bt_d_input_ids, 
                "bt_attention_mask":bt_d_attention_mask}
    
    def forward_pass(test_loader, encoder, has_qrel_label_sample_ids):
        embeddings = []
        eid2sid = []    # embedding idx to sample id
        encoder.zero_grad()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                encoder.eval()
                bt_sample_ids = batch["bt_sample_ids"]
                bt_input_ids = batch['bt_input_ids'].to(args.device)
                bt_attention_mask = batch['bt_attention_mask'].to(args.device)
                embs = encoder(bt_input_ids, bt_attention_mask)
                embs = embs.detach().cpu().numpy()
                
                
                sifted_sample_ids = []
                sifted_embs = []
                for i in range(len(bt_sample_ids)):
                    if bt_sample_ids[i] not in has_qrel_label_sample_ids:
                        continue
                    sifted_sample_ids.append(bt_sample_ids[i])
                    sifted_embs.append(embs[i].reshape(1, -1))
                
                if len(sifted_embs) > 0:
                    sifted_embs = np.concatenate(sifted_embs)
                    embeddings.append(sifted_embs)
                    eid2sid.extend(sifted_sample_ids)
                else:
                    continue

            embeddings = np.concatenate(embeddings, axis = 0)
        
        torch.cuda.empty_cache()
        return embeddings, eid2sid
    
    # for ANCE, query and doc encoders are shared.
    tokenizer, encoder = load_dense_retriever("ANCE", "query", args.retriever_path)
    encoder = encoder.to(args.device)
    
    with open(args.eval_file_path, "r") as f:
        data = f.readlines()
    
    query_encoding_dataset, response_encoding_dataset = [], []
    n_query_candidate, n_response_candidate = 0, 0
    for line in data:
        record = json.loads(line)
        sample_id = record['sample_id']
        if args.include_query:
            query_list = record[args.eval_field_name]
            if isinstance(query_list, str):
                query_list = [query_list]
            n_query_candidate = len(query_list)   # all line's query_list has the same length
            for query in query_list:
                query_encoding_dataset.append([sample_id, query])
            
        if args.include_response:
            response_list = record['predicted_response']
            if isinstance(response_list, str):
                response_list = [response_list]
            n_response_candidate = len(response_list)
            for response in response_list:
                response_encoding_dataset.append([sample_id, response])

    has_qrel_label_sample_ids = get_has_qrel_label_sample_ids(args.qrel_file_path)
    if args.include_query:
        query_test_loader = DataLoader(query_encoding_dataset, batch_size = 32, shuffle=False, collate_fn=query_encoding_collate_fn)
        query_embeddings, query_eid2sid = forward_pass(query_test_loader, encoder, has_qrel_label_sample_ids)
                
    if args.include_response:
        response_test_loader = DataLoader(response_encoding_dataset, batch_size = 32, shuffle=False, collate_fn=response_encoding_collate_fn)
        response_embeddings, response_eid2sid = forward_pass(response_test_loader, encoder, has_qrel_label_sample_ids)
    
    # filter out duplicate sample_ids
    eid2sid = query_eid2sid if query_eid2sid else response_eid2sid
    new_eid2sid = []
    eid2sid_set = set()
    for x in eid2sid:
        if x not in eid2sid_set:
            new_eid2sid.append(x)
            eid2sid_set.add(x)
    eid2sid = new_eid2sid
    
    torch.cuda.empty_cache()
    
    # Different cases. We finally return one emebdding for each sample_id.
    if n_query_candidate == 1 and n_response_candidate == 1:
        return (query_embeddings + response_embeddings) / 2, eid2sid  
    elif n_query_candidate >= 1 and n_response_candidate > 1:
        query_embeddings = query_embeddings.reshape(query_embeddings.shape[0] // n_query_candidate, n_query_candidate, query_embeddings.shape[1])
        response_embeddings = response_embeddings.reshape(response_embeddings.shape[0] // n_response_candidate, n_response_candidate, response_embeddings.shape[1])
        if args.aggregation_method == "maxprob":
            embeddings = (query_embeddings[:, 0, :] + response_embeddings[:, 0, :]) / 2
            return embeddings, eid2sid
        elif args.aggregation_method == "mean":
            embeddings = np.concatenate([query_embeddings, response_embeddings], axis = 1).mean(axis=1)
            return embeddings, eid2sid
        elif args.aggregation_method == "sc":
            if n_query_candidate == 1:
                query_embeddings = query_embeddings[:, 0, :]
                response_embeddings, _ = batch_closest_candidate(response_embeddings)
            else:
                query_embeddings, response_embeddings = batch_closest_candidate(query_embeddings, response_embeddings)
            return (query_embeddings + response_embeddings) / 2, eid2sid  
        else:
            raise NotImplementedError
    elif n_response_candidate == 0: # only query (rewrite)
        query_embeddings = query_embeddings.reshape(query_embeddings.shape[0] // n_query_candidate, n_query_candidate, query_embeddings.shape[1])
        if args.aggregation_method == "maxprob":
            query_embeddings = query_embeddings[:, 0, :]
        elif args.aggregation_method == "mean":
            query_embeddings = np.mean(query_embeddings, axis=1)
        elif args.aggregation_method == "sc":
            query_embeddings, _ = batch_closest_candidate(query_embeddings)
        else:
            raise NotImplementedError
        return query_embeddings, eid2sid
    else:   # only response
        response_emebddings = response_embeddings.reshape(response_embeddings.shape[0] // n_response_candidate, n_response_candidate, response_embeddings.shape[1])
        if args.aggregation_method == "maxprob":
            response_emebddings = response_emebddings[:, 0, :]
        elif args.aggregation_method == "mean":
            response_embeddings = np.mean(response_embeddings, axis=1)
        elif args.aggregation_method == "sc":
            response_embeddings, _ = batch_closest_candidate(response_embeddings)
        return response_emebddings, eid2sid



def batch_closest_candidate(embeddings, affiliated_embeddings=None):
    has_aff = False
    if affiliated_embeddings is not None:
        has_aff = True
        
    res = []
    res_aff = []    # corresponding affiliated_embeddings of embeddings.
    for i in range(embeddings.shape[0]):
        # Calculate the dot product of all pairs of embeddings in the batch
        dot_products = np.dot(embeddings[i], embeddings[i].T)

        # Calculate the sum of each row to get the total dot product for each candidate
        candidate_dots = np.sum(dot_products, axis=1)

        # Find the index of the candidate with the highest total dot product
        closest_idx = np.argmax(candidate_dots)

        # Return the embedding for the closest candidate
        res.append(embeddings[i][closest_idx].reshape(1, -1))

        if has_aff:
            res_aff.append(affiliated_embeddings[i][closest_idx].reshape(1, -1))

    return np.concatenate(res, axis=0), np.concatenate(res_aff, axis=0) if has_aff else None

def faiss_flat_retrieval_one_by_one_and_finally_merge(args, query_embs):
    index = build_faiss_index(args)
    merged_candidate_matrix = None
    
    # Automaticall get the number of doc blocks
    args.num_doc_block = 1
    for filename in os.listdir(args.index_path):
        try:
            args.num_doc_block = max(args.num_doc_block, int(filename.split(".")[1]))
        except:
            continue
    args.num_doc_block += 1
    print("Automatically detect that the number of doc blocks is: {}".format(args.num_doc_block))
    
    for block_id in range(args.num_doc_block):
        logger.info("Loading doc block " + str(block_id))

        # load doc embeddings
        with open(os.path.join(args.index_path, "doc_emb_block.{}.pb".format(block_id)), 'rb') as handle:
            cur_doc_embs = pickle.load(handle)
        with open(os.path.join(args.index_path, "doc_embid_block.{}.pb".format(block_id)), 'rb') as handle:
            cur_eid2did = pickle.load(handle)
            if isinstance(cur_eid2did, list):
                cur_eid2did = np.array(cur_eid2did)

        # Split to avoid the doc embeddings to be too large
        num_total_doc_per_block = len(cur_doc_embs)
        num_doc_per_split = 500000    # please set it according to your GPU size. 700w doc needs ~28GB
        num_split_block = max(1, num_total_doc_per_block // num_doc_per_split)
        logger.info("num_total_doc: {}".format(num_total_doc_per_block))
        logger.info("num_doc_per_split: {}".format(num_doc_per_split))
        logger.info("num_split_block: {}".format(num_split_block))
        cur_doc_embs_list = np.array_split(cur_doc_embs, num_split_block)
        cur_eid2did_list = np.array_split(cur_eid2did, num_split_block)
        for split_idx in range(len(cur_doc_embs_list)):
            cur_doc_embs = cur_doc_embs_list[split_idx]
            cur_eid2did = cur_eid2did_list[split_idx]
            logger.info("Adding block {} split {} into index...".format(block_id, split_idx))
            index.add(cur_doc_embs)
            
            # ann search
            tb = time.time()

            D, I = index.search(query_embs, args.top_n)
            elapse = time.time() - tb
            logger.info({
                'time cost': elapse,
                'query num': query_embs.shape[0],
                'time cost per query': elapse / query_embs.shape[0]
            })

            candidate_did_matrix = cur_eid2did[I] # doc embedding_idx -> real doc id
            D = D.tolist()
            candidate_did_matrix = candidate_did_matrix.tolist()
            candidate_matrix = []

            for score_list, doc_list in zip(D, candidate_did_matrix):
                candidate_matrix.append([])
                for score, doc in zip(score_list, doc_list):
                    candidate_matrix[-1].append((score, doc))
                assert len(candidate_matrix[-1]) == len(doc_list)
            assert len(candidate_matrix) == I.shape[0]

            index.reset()
            del cur_doc_embs
            del cur_eid2did

            if merged_candidate_matrix == None:
                merged_candidate_matrix = candidate_matrix
                continue
            
            # Merge
            merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
            merged_candidate_matrix = []
            for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                            candidate_matrix):
                p1, p2 = 0, 0
                merged_candidate_matrix.append([])
                while p1 < args.top_n and p2 < args.top_n:
                    if merged_list[p1][0] >= cur_list[p2][0]:
                        merged_candidate_matrix[-1].append(merged_list[p1])
                        p1 += 1
                    else:
                        merged_candidate_matrix[-1].append(cur_list[p2])
                        p2 += 1
                while p1 < args.top_n:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                while p2 < args.top_n:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1

    merged_D, merged_I = [], []

    for merged_list in merged_candidate_matrix:
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)

    logger.info(merged_D.shape)
    logger.info(merged_I.shape)

    return merged_D, merged_I

def dense_retrieval(args):
    query_embs, eid2sid = get_embeddings(args)
    score_mat, did_mat = faiss_flat_retrieval_one_by_one_and_finally_merge(args, query_embs)
    
    # write to file
    run_trec_file = os.path.join(args.retrieval_output_path, "res.trec")
    with open(run_trec_file, "w") as f:
        for eid in range(len(did_mat)):
            sample_id = eid2sid[eid]
            retrieval_scores = score_mat[eid]
            retrieved_dids = did_mat[eid]
            for i in range(len(retrieval_scores)):
                rank = i + 1
                doc_id = retrieved_dids[i]
                rank_score = args.top_n - i # use the rank score for pytrec
                real_score = retrieval_scores[i] 
                f.write("{} {} {} {} {} {} {}".format(sample_id, "Q0", doc_id, rank, rank_score, real_score, "ance"))
                f.write('\n')
            
    # evaluation
    trec_eval(run_trec_file, args.qrel_file_path, args.retrieval_output_path, args.rel_threshold)
    


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_file_path", type=str, required=True)
    parser.add_argument("--eval_field_name", type=str, required=True, help="Field name of the rewrite in the eval_file. E.g., t5_rewrite")
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--qrel_file_path", type=str, required=True)
    parser.add_argument("--retriever_path", type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=64, help="Max single query length")
    parser.add_argument("--aggregation_method", type=str, default="maxprob", choices=["sc", "mean", "maxprob"])
    
    parser.add_argument("--include_query", action="store_true")
    parser.add_argument("--include_response", action="store_true")
    
    parser.add_argument("--use_gpu_in_faiss", action="store_true", help="whether to use gpu in faiss or not.")
    parser.add_argument("--n_gpu_for_faiss", type=int, default=1, help="should be set if use_gpu_in_faiss")
    
    
    parser.add_argument("--top_n", type=int, default=1000)
    parser.add_argument("--rel_threshold", type=int, required=True, help="CAsT-20: 2, Others: 1")

    parser.add_argument("--retrieval_output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # main
    args = parser.parse_args()
    assert args.include_query or args.include_response
    os.makedirs(args.retrieval_output_path, exist_ok=True)
    with open(os.path.join(args.retrieval_output_path, "parameters.txt"), "w") as f:
        params = vars(args)
        f.write(json.dumps(params, indent=4))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    logger.info("---------------------The arguments are:---------------------")
    pprint(args)

    return args



if __name__ == '__main__':
    args = get_args()
    set_seed(args) 
    dense_retrieval(args)
    logger.info("Dense retrieval and evaluation finish!")