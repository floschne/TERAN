from __future__ import print_function

import time

import numpy
import numpy as np
import torch
import tqdm

from evaluate_utils.dcg import DCG
from models.loss import order_sim, AlignmentContrastiveLoss
from utils import get_model, AverageMeter, LogCollector
from data import get_coco_image_retrieval_data, get_test_loader


def encode_data(model, data_loader, log_step=10, logging=print, max_cap_len=None, max_img_len=None):
    """
    Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    img_lengths = []
    cap_lengths = []

    if max_cap_len is None or max_img_len is None:
        # compute maximum lenghts in the whole dataset
        # max_cap_len = 88
        # max_img_len = 37
        max_cap_len = -1
        max_img_len = -1
        for _, _, img_length, cap_length, _, _ in tqdm.tqdm(data_loader, desc="Finding maximum image and caption length..."):
            max_cap_len = max(max_cap_len, max(cap_length))
            max_img_len = max(max_img_len, max(img_length))
        print(f"Found max image and cap lengths: {max_cap_len} , {max_img_len}")

    for i, (images, targets, img_length, cap_length, boxes, ids) in enumerate(tqdm.tqdm(data_loader, desc='Encoding Data for evaluation...')):
        # make sure val logger is used
        model.logger = val_logger

        if type(targets) == tuple or type(targets) == list:
            captions, features, wembeddings = targets
            # captions = features  # Very weird, I know
            text = features
        else:
            text = targets
            captions = targets

        # compute the embeddings
        with torch.no_grad():
            _, _, img_emb, cap_emb, cap_length = model.forward_emb(images, text, img_length, cap_length, boxes)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None: # N x max_len x 1024
                img_embs = torch.zeros((len(data_loader.dataset), max_img_len, img_emb.size(2)))
                cap_embs = torch.zeros((len(data_loader.dataset), max_cap_len, cap_emb.size(2)))

            # preserve the embeddings by copying from gpu and converting to numpy
            img_embs[ids, :img_emb.size(0), :] = img_emb.cpu().permute(1, 0, 2)
            cap_embs[ids, :cap_emb.size(0), :] = cap_emb.cpu().permute(1, 0, 2)
            img_lengths.extend(img_length)
            cap_lengths.extend(cap_length)

            # measure accuracy and record loss
            # model.forward_loss(None, None, img_emb, cap_emb, img_length, cap_length)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions

    # p = np.random.permutation(len(data_loader.dataset) // 5) * 5
    # p = np.transpose(np.tile(p, (5, 1)))
    # p = p + np.array([0, 1, 2, 3, 4])
    # p = p.flatten()
    # img_embs = img_embs[p]
    # cap_embs = cap_embs[p]

    return img_embs, cap_embs, img_lengths, cap_lengths


def evalrank(config, checkpoint, split='dev', fold5=False, eval_t2i=True, eval_i2t=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    evalrank_start_time = time.time()

    # load model and options
    # checkpoint = torch.load(model_path)
    data_path = config['dataset']['data']
    measure = config['training']['measure']

    # construct model
    model = get_model(config)

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)

    print('Loading dataset')
    data_loader = get_test_loader(config, workers=4, split_name=split)

    # initialize ndcg scorer
    ndcg_val_scorer = DCG(config, len(data_loader.dataset), split, rank=25, relevance_methods=['rougeL', 'spice'])

    # initialize similarity matrix evaluator
    sim_matrix_fn = AlignmentContrastiveLoss(aggregation=config['training']['alignment-mode'],
                                             return_similarity_mat=True) if config['training'][
                                                                                'loss-type'] == 'alignment' else None

    print('Computing results...')
    encode_data_start_time = time.time()
    img_embs, cap_embs, img_lenghts, cap_lenghts = encode_data(model, data_loader)
    print(f"Time elapsed for encode_data: {time.time() - encode_data_start_time} seconds.")

    torch.cuda.empty_cache()

    # if checkpoint2 is not None:
    #     # construct model
    #     model2 = get_model(config2)
    #     # load model state
    #     model2.load_state_dict(checkpoint2['model'], strict=False)
    #     img_embs2, cap_embs2 = encode_data(model2, data_loader)
    #     print('Using 2-model ensemble')
    # else:
    #     img_embs2, cap_embs2 = None, None
    #     print('Using NO ensemble')

    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        if eval_i2t:
            eval_i2t_start_time = time.time()

            r, rt = i2t(img_embs,
                        cap_embs,
                        img_lenghts,
                        cap_lenghts,
                        return_ranks=True,
                        ndcg_scorer=ndcg_val_scorer,
                        sim_function=sim_matrix_fn,
                        cap_batches=5)
            ar = (r[0] + r[1] + r[2]) / 3
            print("Average i2t Recall: %.1f" % ar)
            print("Image to text: %.1f %.1f %.1f %.1f %.1f, ndcg_rouge=%.4f, ndcg_spice=%.4f" % r)

            print(f"Time elapsed for i2t evaluation without 5-fold CV: {time.time() - eval_i2t_start_time} seconds.")

        if eval_t2i:
            eval_t2i_start_time = time.time()

            ri, rti = t2i(img_embs,
                          cap_embs,
                          img_lenghts,
                          cap_lenghts,
                          return_ranks=True,
                          ndcg_scorer=ndcg_val_scorer,
                          sim_function=sim_matrix_fn,
                          im_batches=5)

            ari = (ri[0] + ri[1] + ri[2]) / 3
            print("Average t2i Recall: %.1f" % ari)
            print("Text to image: %.1f %.1f %.1f %.1f %.1f, ndcg_rouge=%.4f, ndcg_spice=%.4f" % ri)

            print(f"Time elapsed for t2i evaluation without 5-fold CV: {time.time() - eval_t2i_start_time} seconds.")

        if eval_i2t and eval_t2i:
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f" % rsum)



    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            if eval_i2t:
                r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) * 5000],
                             img_lenghts[i * 5000:(i + 1) * 5000], cap_lenghts[i * 5000:(i + 1) * 5000],
                             return_ranks=True, ndcg_scorer=ndcg_val_scorer, fold_index=i, sim_function=sim_matrix_fn,
                             cap_batches=1)
                print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f ndcg_spice=%.4f" % r)
                if i == 0:
                    rt = rt0
                ar = (r[0] + r[1] + r[2]) / 3
            if eval_t2i:
                ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) * 5000],
                               img_lenghts[i * 5000:(i + 1) * 5000], cap_lenghts[i * 5000:(i + 1) * 5000],
                               return_ranks=True, ndcg_scorer=ndcg_val_scorer, fold_index=i, sim_function=sim_matrix_fn,
                               im_batches=1)
                print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f, ndcg_spice=%.4f" % ri)
                if i == 0:
                    rti = rti0
                ari = (ri[0] + ri[1] + ri[2]) / 3


            if eval_t2i and eval_i2t:
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            elif eval_t2i:
                print("ari: %.1f" % (ari,))
            elif eval_i2t:
                print("ar: %.1f" % (ar,))

            if eval_t2i and eval_i2t:
                results += [list(r) + list(ri) + [ar, ari, rsum]]  # 7 + 7 + 3 = 17 elements
            elif eval_t2i:
                results += [list(ri) + [ari]]  # 7 + 1 = 8 elements
            elif eval_i2t:
                results += [list(r) + [ar]]  # 7 + 1 = 8 elements

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        if eval_t2i and eval_i2t:
            print("rsum: %.1f" % (mean_metrics[16] * 6))
            print("Average i2t Recall: %.1f" % mean_metrics[14])
            print("Image to text: %.1f %.1f %.1f %.1f %.1f ndcg_rouge=%.4f ndcg_spice=%.4f" %
                  mean_metrics[:7])
            print("Average t2i Recall: %.1f" % mean_metrics[15])
            print("Text to image: %.1f %.1f %.1f %.1f %.1f ndcg_rouge=%.4f ndcg_spice=%.4f" %
                  mean_metrics[7:14])
        elif eval_t2i:
            print("Average t2i Recall: %.1f" % mean_metrics[7])
            print("Text to image: %.1f %.1f %.1f %.1f %.1f ndcg_rouge=%.4f ndcg_spice=%.4f" %
                  mean_metrics[:7])
        elif eval_i2t:
            print("Average i2t Recall: %.1f" % mean_metrics[7])
            print("Image to text: %.1f %.1f %.1f %.1f %.1f ndcg_rouge=%.4f ndcg_spice=%.4f" %
                  mean_metrics[:7])




    if eval_t2i and eval_i2t:
        torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')
    elif eval_t2i:
        torch.save({'rti': rti}, 'ranks.pth.tar')
    elif eval_i2t:
        torch.save({'rt': rt}, 'ranks.pth.tar')

    print(f"Time elapsed for evalrank(): {time.time() - evalrank_start_time} seconds.")


def i2t(images, captions, img_lenghts, cap_lenghts, npts=None, return_ranks=False, ndcg_scorer=None, fold_index=0,
        measure='dot', sim_function=None, cap_batches=1, wicsmmir=False):
    """
    Images->Text (Image Annotation)
    Images: (num_captions_per_image * N, K) matrix of images
    Captions: (num_captions_per_image * N, K) matrix of captions
    """
    if not wicsmmir:
        num_captions_per_image = 5
    else:
        num_captions_per_image = 1
    if npts is None:
        npts = images.shape[0] // num_captions_per_image
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    rougel_ndcgs = numpy.zeros(npts)
    spice_ndcgs = numpy.zeros(npts)
    # captions = captions.cuda()
    captions_per_batch = captions.shape[0] // cap_batches

    for index in tqdm.trange(npts):

        # Get query image
        im = images[num_captions_per_image * index].reshape(1, images.shape[1], images.shape[2])
        im = im.cuda() if sim_function is not None else im
        im_len = [img_lenghts[num_captions_per_image * index]]

        d = None

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], num_captions_per_image * (index + bs))
                im2 = images[num_captions_per_image * index:mx:num_captions_per_image]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            if sim_function is None:
                d = torch.mm(im[:, 0, :], captions[:, 0, :].t())
                d = d.cpu().numpy().flatten()
            else:
                for i in range(cap_batches):
                    captions_now = captions[i * captions_per_batch:(i + 1) * captions_per_batch]
                    cap_lenghts_now = cap_lenghts[i * captions_per_batch:(i + 1) * captions_per_batch]
                    captions_now = captions_now.cuda()

                    d_align = sim_function(im, captions_now, im_len, cap_lenghts_now)
                    d_align = d_align.cpu().numpy().flatten()
                    # d_matching = torch.mm(im[:, 0, :], captions[:, 0, :].t())
                    # d_matching = d_matching.cpu().numpy().flatten()
                    if d is None:
                        d = d_align  # + d_matching
                    else:
                        d = numpy.concatenate([d, d_align], axis=0)

        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(num_captions_per_image * index, num_captions_per_image * index + num_captions_per_image, 1):

            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

        if ndcg_scorer is not None:
            rougel_ndcgs[index], spice_ndcgs[index] = ndcg_scorer.compute_ndcg(npts, index, inds.astype(int),
                                                                               fold_index=fold_index,
                                                                               retrieval='sentence').values()

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    mean_rougel_ndcg = np.mean(rougel_ndcgs[rougel_ndcgs != 0])
    mean_spice_ndcg = np.mean(spice_ndcgs[spice_ndcgs != 0])
    if return_ranks:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg)


def t2i(images, captions, img_lenghts, cap_lenghts, npts=None, return_ranks=False, ndcg_scorer=None, fold_index=0,
        measure='dot', sim_function=None, im_batches=1, wicsmmir=False):
    """
    Text->Images (Image Search)
    Images: (num_captions_per_image * N, K) matrix of images
    Captions: (num_captions_per_image * N, K) matrix of captions
    """
    if not wicsmmir:
        num_captions_per_image = 5
    else:
        num_captions_per_image = 1
    if npts is None:
        npts = images.shape[0] // num_captions_per_image
    ims = torch.stack([images[i] for i in range(0, len(images), num_captions_per_image)], dim=0)
    # ims = ims.cuda()
    ims_len = [img_lenghts[i] for i in range(0, len(images), num_captions_per_image)]

    ranks = numpy.zeros(num_captions_per_image * npts)
    top50 = numpy.zeros((num_captions_per_image * npts, 50))
    rougel_ndcgs = numpy.zeros(num_captions_per_image * npts)
    spice_ndcgs = numpy.zeros(num_captions_per_image * npts)

    images_per_batch = ims.shape[0] // im_batches

    for index in tqdm.trange(npts):

        # Get query captions
        queries = captions[num_captions_per_image * index:num_captions_per_image * index + num_captions_per_image]
        queries = queries.cuda() if sim_function is not None else queries
        queries_len = cap_lenghts[num_captions_per_image * index:num_captions_per_image * index + num_captions_per_image]

        d = None

        # Compute scores
        if measure == 'order':
            bs = 100
            if num_captions_per_image * index % bs == 0:
                mx = min(captions.shape[0], num_captions_per_image * index + bs)
                q2 = captions[num_captions_per_image * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (num_captions_per_image * index) % bs:(num_captions_per_image * index) % bs + num_captions_per_image].T
        else:
            if sim_function is None:
                d = torch.mm(queries[:, 0, :], ims[:, 0, :].t())
                d = d.cpu().numpy()
            else:
                for i in range(im_batches):
                    ims_now = ims[i * images_per_batch:(i + 1) * images_per_batch]
                    ims_len_now = ims_len[i * images_per_batch:(i + 1) * images_per_batch]
                    ims_now = ims_now.cuda()

                    # d = numpy.dot(queries, ims.T)
                    # d_align is the (MrSw) aggregated/pooled similarity matrix A in the paper
                    d_align = sim_function(ims_now, queries, ims_len_now, queries_len).t()
                    d_align = d_align.cpu().numpy()
                    # d_matching = torch.mm(queries[:, 0, :], ims[:, 0, :].t())
                    # d_matching = d_matching.cpu().numpy()
                    if d is None:
                        d = d_align  # + d_matching
                    else:
                        d = numpy.concatenate([d, d_align], axis=1)

        # d contains all aggregated/pooled similarity matrices for all query-image pairs in the test set
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            # in che posizione e' l'immagine (index) che ha questa caption (num_captions_per_image*index + i)
            ranks[num_captions_per_image * index + i] = tmp
            top50[num_captions_per_image * index + i] = inds[i][0:50]
            # calculate ndcg
            if ndcg_scorer is not None:
                rougel_ndcgs[num_captions_per_image * index + i], spice_ndcgs[num_captions_per_image * index + i] = \
                    ndcg_scorer.compute_ndcg(npts, num_captions_per_image * index + i, inds[i].astype(int),
                                             fold_index=fold_index, retrieval='image').values()

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    mean_rougel_ndcg = np.mean(rougel_ndcgs)
    mean_spice_ndcg = np.mean(spice_ndcgs)

    if return_ranks:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg), (ranks, top50)
    else:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg)

