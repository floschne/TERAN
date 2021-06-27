import argparse
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import repeat

import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer
from typing import List

# noinspection PyUnresolvedReferences
from data import get_image_retrieval_data, QueryEncoder, WICSMMIRImageRetrievalDataset, CocoImageRetrievalDatasetBase, \
    FlickrImageRetrievalDatasetBase
from models.loss import AlignmentContrastiveLoss
from models.teran import TERAN
from utils import AverageMeter, LogCollector


def persist_img_embs(config, data_loader, dataset_indices, numpy_img_emb, pool):
    dst_root = Path(os.getcwd()).joinpath(config['image-retrieval']['pre_computed_img_embeddings_root'])
    if not dst_root.exists():
        dst_root.mkdir(parents=True, exist_ok=True)

    assert len(dataset_indices) == len(numpy_img_emb)
    img_ids = get_image_ids(dataset_indices, data_loader)

    futures = []
    for idx in range(len(img_ids)):
        dst = dst_root.joinpath(img_ids[idx] + '.npz')
        if dst.exists():
            continue
        futures.append(pool.submit(np.savez_compressed, file=str(dst), img_emb=numpy_img_emb[idx]))

    for f in as_completed(futures):
        pass


def encode_data_for_inference(model: TERAN, data_loader, log_step=10, logging=print, pre_compute_img_embs=False,
                              persist_pool=None):
    # compute the embedding vectors v_i, s_j (paper) for each image region and word respectively
    # -> forwarding the data through the respective TE stacks
    print(
        f'{"Pre-" if pre_compute_img_embs else ""}Computing image {"" if pre_compute_img_embs else "and query "}embeddings...')

    # we don't need autograd for inference
    model.eval()

    # array to keep all the embeddings
    # TODO maybe we can store those embeddings in an index and load it instead of computing each time for each query
    query_embs = None
    num_query_feats = None
    num_img_feats = None  # all images have a fixed size of pre-extracted features of 36 + 1 regions
    img_embs = None

    # make sure val logger is used
    batch_time = AverageMeter()
    val_logger = LogCollector()
    model.logger = val_logger

    start_time = time.time()
    for i, (img_feature_batch, img_feat_bboxes_batch, img_feat_len_batch, query_token_batch, query_len_batch,
            dataset_indices) in enumerate(tqdm(data_loader)):
        batch_start_time = time.time()
        """
        the data loader returns None values for the respective batches if the only query was already loaded 
        -> query_token_batch, query_len_batch = None, None
        """

        with torch.no_grad():
            # compute the query embedding only in the first iteration (also because there is only 1 query in IR)
            if query_embs is None and not pre_compute_img_embs:
                # TODO maybe we can get the most matching roi from query_emb_aggr?
                query_emb_aggr, query_emb, _ = model.forward_txt(query_token_batch, query_len_batch)

                # store results as np arrays for further processing or persisting
                num_query_feats = query_len_batch[0] if isinstance(query_len_batch, list) else query_len_batch
                query_feat_dim = query_emb.size(2)
                query_embs = torch.zeros((1, num_query_feats, query_feat_dim))
                query_embs[0, :, :] = query_emb.cpu().permute(1, 0, 2)

            # compute every image embedding in the dataset
            img_emb_aggr, img_emb = model.forward_img(img_feature_batch, img_feat_len_batch, img_feat_bboxes_batch)

            # init array to store results for further processing or persisting
            if img_embs is None:
                num_img_feats = img_feat_len_batch[0] if isinstance(img_feat_len_batch,
                                                                    list) else img_feat_len_batch
                img_feat_dim = img_emb.size(2)
                img_embs = torch.zeros((len(data_loader.dataset), num_img_feats, img_feat_dim))

            numpy_img_emb = img_emb.cpu().permute(1, 0, 2)  # why are we permuting here? -> TERAN
            img_embs[dataset_indices, :, :] = numpy_img_emb
            if pre_compute_img_embs:
                # if we are in a pre-compute run, persist the arrays
                persist_img_embs(model_config, data_loader, dataset_indices, numpy_img_emb, persist_pool)

        # measure elapsed time per batch
        batch_time.update(time.time() - batch_start_time)

        if i % log_step == 0:
            logging(
                f"Batch: [{i}/{len(data_loader)}]\t{str(model.logger)}\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})")
        del img_feature_batch, query_token_batch

    print(
        f"Time elapsed to {'encode' if not pre_compute_img_embs else 'encode and persist'} data: {time.time() - start_time} seconds.")
    return img_embs, query_embs, num_img_feats, num_query_feats


def get_tokenizer(config) -> BertTokenizer:
    return BertTokenizer.from_pretrained(config['text-model']['pretrain'])


def compute_distance_task(sim_matrix_fn, img_embs_batch, query_emb_batch, img_embs_length_batch, query_length_batch):
    # compute and pool the similarity matrices to get the global distance between the image and the query
    alignment_distance, alignment_matrix = sim_matrix_fn(img_embs_batch, query_emb_batch, img_embs_length_batch,
                                                         query_length_batch)

    return alignment_distance, alignment_matrix


def compute_distances(img_embs,
                      query_embs,
                      img_lengths,
                      query_lengths,
                      config,
                      return_wra_matrices=True,
                      dist_pool=None):
    # necessary for multi processing many files / objects
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

    # initialize similarity matrix evaluator
    sim_matrix_fn = AlignmentContrastiveLoss(aggregation=config['image-retrieval']['alignment_mode'],
                                             return_aggregated_similarity_mat=True,
                                             return_alignment_mat=True)

    img_embs_per_batch = config['image-retrieval']['batch_size']
    num_imgs = img_embs.size(0)
    img_emb_batches = num_imgs // img_embs_per_batch
    img_embs_per_batch = [img_embs_per_batch] * img_emb_batches

    # batch the image embs so that the last batch contains the remaining num_imgs % img_embs_per_batch images
    if num_imgs % img_embs_per_batch[0] != 0:
        img_embs_per_batch = img_embs_per_batch + [num_imgs % img_embs_per_batch[0]]
        img_emb_batches += 1

    # distances storage
    distances = None

    # wra matrices storage
    matrices = None

    # since its always the same query we can reuse the batch
    # (TODO maybe we can even just use a batch of size 1?! -> check the sim_matrix_fn)
    query_emb_batch = query_embs[:1]
    query_length_batch = [query_lengths[0] if isinstance(query_lengths, list) else query_lengths for _ in range(1)]

    start_time = time.time()
    # batch-wise compute the alignment distance between the images and the query
    with tqdm(total=img_emb_batches, desc='Computing distances...') as pbar:
        img_embs_batches = []
        img_embs_length_batches = []
        for i in range(img_emb_batches):
            # get the current batch
            img_embs_batch = img_embs[i * img_embs_per_batch[i]:(i + 1) * img_embs_per_batch[i]]
            img_embs_batches.append(img_embs_batch)
            img_embs_length_batch = [img_lengths for _ in range(img_embs_per_batch[i])]
            img_embs_length_batches.append(img_embs_length_batch)

        # # compute and pool the similarity matrices to get the global distance between the image and the query

        # compute in parallel if worker pool is provided # todo use ray
        if dist_pool is not None:
            # !!! we have to use
            # map here to preserve the ordering so that the indices match the image ids !!!
            dists = dist_pool.map(compute_distance_task,
                                  repeat(sim_matrix_fn),
                                  img_embs_batches,
                                  repeat(query_emb_batch),
                                  img_embs_length_batches,
                                  repeat(query_length_batch))
        else:
            # compute serially
            dists = [compute_distance_task(sim_matrix_fn,
                                           img_emb_batch,
                                           query_emb_batch,
                                           img_emb_batch_len,
                                           query_length_batch)
                     for img_emb_batch, img_emb_batch_len in zip(img_embs_batches, img_embs_length_batches)]

        for alignment_distance, alignment_matrix in dists:
            alignment_distance = alignment_distance.squeeze().numpy()
            alignment_matrix = alignment_matrix.squeeze().numpy()

            # store the distances
            if distances is None:
                distances = alignment_distance
            else:
                distances = np.concatenate([distances, alignment_distance], axis=0)

            if return_wra_matrices:
                # store matrices
                if matrices is None:
                    matrices = alignment_matrix
                else:
                    matrices = np.concatenate([matrices, alignment_matrix], axis=0)
            pbar.update(1)

    print(f"Time elapsed to compute and pool the similarity matrices: {time.time() - start_time} seconds.")
    if return_wra_matrices:
        return distances, matrices
    else:
        return distances


# TODO hardcoded is ugly and this is actually not mentioned anywhere due to laziness...
def build_coco_img_id(coco_id: int, prefix: str = 'COCO_'):
    res = str(coco_id)
    # coco image ids are filled with 0 until it's a 12 char string
    while len(res) != 12:
        res = '0' + res

    return prefix + res


def get_image_ids(dataset_indices, dataloader) -> List[str]:
    # WICSMMIR
    if isinstance(dataloader.dataset, WICSMMIRImageRetrievalDataset):
        imgs = dataloader.dataset.dataframe.iloc[dataset_indices]['wikicaps_id'].to_numpy().tolist()
        return [str(i) for i in imgs]

    # COCO
    if isinstance(dataloader.dataset, CocoImageRetrievalDatasetBase):
        return [build_coco_img_id(dataloader.dataset.get_image_metadata(idx)[0]) for idx in dataset_indices]
    elif isinstance(dataloader, CocoImageRetrievalDatasetBase):
        return [build_coco_img_id(dataloader.get_image_metadata(idx)[0]) for idx in dataset_indices]

    # F30k
    if isinstance(dataloader.dataset, FlickrImageRetrievalDatasetBase):
        return dataloader.dataset.get_image_ids(dataset_indices)
    else:
        raise NotImplementedError("Only COCO, F30k, and WICSMMIR are supported!")


def load_precomputed_image_embeddings(config, num_workers):
    print("Loading pre-computed image embeddings...")
    start = time.time()

    # returns a PreComputedCocoImageEmbeddingsDataset
    dataset = get_image_retrieval_data(config, num_workers=num_workers)

    # get the img embeddings and convert them to Tensors
    np_img_embs = np.array(list(dataset.img_embs.values()))
    img_embs = torch.Tensor(np_img_embs)
    img_lengths = len(np_img_embs[0])
    print(f"Time elapsed to load pre-computed embeddings: {time.time() - start} seconds!")
    return img_embs, img_lengths, dataset


def load_teran(config, checkpoint):
    # construct model
    model = TERAN(config)
    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)
    return model


def top_k_image_retrieval(opts, config, checkpoint) -> List[str]:
    model = load_teran(config, checkpoint)

    use_precomputed_img_embeddings = config['image-retrieval']['use_precomputed_img_embeddings']
    if use_precomputed_img_embeddings:
        # load pre computed img embs
        img_embs, img_lengths, dataset = load_precomputed_image_embeddings(config, num_workers=opts.num_data_workers)
        # compute query emb
        query_encoder = QueryEncoder(config, model)
        query_embs, query_lengths = query_encoder.compute_query_embedding(opts.query)

    else:
        # returns a Dataloader of a PreComputedCocoFeaturesDataset
        dataset = get_image_retrieval_data(config,
                                           query=opts.query,
                                           num_workers=opts.num_data_workers)
        # encode the data (i.e. compute the embeddings / TE outputs for the images and query)
        img_embs, query_embs, img_lengths, query_lengths = encode_data_for_inference(model, dataset)

    if opts.device == "cuda":
        torch.cuda.empty_cache()

    print(f"Images Embeddings: {img_embs.shape[0]}, Query Embeddings: {query_embs.shape[0]}")

    # compute the matching scores
    distance_sorted_indices = compute_distances(img_embs, query_embs, img_lengths, query_lengths, config)
    top_k_indices = distance_sorted_indices[:opts.top_k]

    # get the image names
    top_k_images = get_image_ids(top_k_indices, dataloader=dataset)
    return top_k_images


def prepare_model_checkpoint_and_config(opts):
    checkpoint = torch.load(opts.model, map_location=torch.device(opts.device))
    print('Checkpoint loaded from {}'.format(opts.model))
    model_checkpoint_config = checkpoint['config']

    with open(opts.config, 'r') as yml_file:
        loaded_config = yaml.load(yml_file)
        # Override some mandatory things in the configuration
        model_checkpoint_config['dataset'] = loaded_config['dataset']
        model_checkpoint_config['image-retrieval'] = loaded_config['image-retrieval']

    return model_checkpoint_config, checkpoint


def pre_compute_img_embeddings(opts, config, checkpoint):
    # construct model
    model = TERAN(config)

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)

    print('Loading dataset')
    data_loader = get_image_retrieval_data(config,
                                           query=opts.query,
                                           num_workers=opts.num_data_workers,
                                           pre_compute_img_embs=True)
    with ProcessPoolExecutor(max_workers=32) as persist_pool:
        # encode the data (i.e. compute the embeddings / TE outputs for the images and query)
        encode_data_for_inference(model, data_loader, pre_compute_img_embs=True, persist_pool=persist_pool)


if __name__ == '__main__':
    print("CUDA_VISIBLE_DEVICES: " + os.getenv("CUDA_VISIBLE_DEVICES", "NOT SET - ABORTING"))
    if os.getenv("CUDA_VISIBLE_DEVICES", None) is None:
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help="Model (checkpoint) to load. E.g. pretrained_models/coco_MrSw.pth.tar", required=True)
    parser.add_argument('--pre_compute_img_embeddings', action='store_true', help="If set or true, the image "
                                                                                  "embeddings get precomputed and "
                                                                                  "persisted at the directory "
                                                                                  "specified in the config.")
    parser.add_argument('--query', type=str, required='--pre_compute_img_embeddings' not in sys.argv)
    parser.add_argument('--num_data_workers', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--dataset', type=str, choices=['coco', 'wicsmmir', 'f30k'],
                        default='coco')  # TODO support other datasets
    parser.add_argument('--config', type=str, default='configs/teran_coco_MrSw_IR.yaml', help="Which configuration to "
                                                                                              "use for overriding the"
                                                                                              " checkpoint "
                                                                                              "configuration. See "
                                                                                              "into 'config' folder")
    # cpu is only for local test runs
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    opts = parser.parse_args()

    model_config, model_checkpoint = prepare_model_checkpoint_and_config(opts)

    if not opts.pre_compute_img_embeddings:
        top_k_matches = top_k_image_retrieval(opts, model_config, model_checkpoint)
        print(f"##########################################")
        print(f"QUERY: {opts.query}")
        print(f"######## TOP {opts.top_k} RESULTS ########")
        print(top_k_matches)
    else:
        pre_compute_img_embeddings(opts, model_config, model_checkpoint)
