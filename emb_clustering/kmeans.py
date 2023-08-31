# cluster embeddings using faiss

import torch
import numpy as np 
import faiss

import os
import sys
import json
import argparse
import multiprocessing

# append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_json_lines
from utils import rround


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def encode(self, json_line):
        try: 
            data = json.loads(json_line)
            emb = data[args.json_key]
            emb = np.load(emb)['emb']       # (D, N)
            emb = np.transpose(emb).astype('float32')         # (N, D)
            return emb
        except:
            return None


def parser_args():
    parser = argparse.ArgumentParser(description='cluster embeddings using faiss')
    parser.add_argument('--manifest_filepaths_list', 
                        nargs='+', 
                        default=[], 
                        help='space separate listed of keys to extract from json',
                    )
    parser.add_argument('--json_key', type=str, required=True)
    parser.add_argument('--num_clusters', type=int, required=True)
    parser.add_argument('--centroids_save_path', type=str, required=True)
    parser.add_argument('--niter', type=int, default=20)
    parser.add_argument('--nredo', type=int, default=5)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    return args


def main(args):
    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers)

    emb_list = []
    for idx, json_file in enumerate(args.manifest_filepaths_list):
        print(f'Processing file {json_file} {idx + 1}/{len(args.manifest_filepaths_list)}')
        fin = open(json_file, 'r', encoding='utf-8')

        encoded_docs = pool.imap(encoder.encode, fin, 25)
        emb_list.extend(encoded_docs)
    
    print("Number of embeddings: ", len(emb_list))
    # remove None if any present in the list
    emb_list = [emb for emb in emb_list if emb is not None]
    print("Number of embeddings after filtering None: ", len(emb_list))

    emb = np.concatenate(emb_list, axis=0)      # (N, D)
    print(emb.shape)

    max_points_per_centroid = (emb.shape[0] + args.num_clusters - 1) // args.num_clusters
    kmeans = faiss.Kmeans(d=emb.shape[1], 
                          k=args.num_clusters,
                          niter=args.niter,
                          update_index=True,
                          verbose=True,
                          gpu=True,
                          nredo=args.nredo,
                          max_points_per_centroid=max_points_per_centroid,
                          seed=42,
                          )
    
    
    kmeans.train(emb)
    print(kmeans.obj)
    print(kmeans.iteration_stats)

    centroids = kmeans.centroids
    centroids = centroids.astype(np.float32)
    np.savez(args.centroids_save_path, centroids=centroids)
    

if __name__ == '__main__':
    args = parser_args()
    print(args.manifest_filepaths_list)
    main(args)
