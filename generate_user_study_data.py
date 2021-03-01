import argparse
import os
import tarfile as tar
from typing import List

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from data import WICSMMIRDataset


def load_top50(opts):
    ranks = torch.load(opts.ranks_file)
    top50 = ranks['rti'][1]
    return top50


def get_wicsmmir_data(ds_idx: int, dataset: WICSMMIRDataset):
    wikicaps_id, caption = dataset.captions.iloc[ds_idx, 0:2]
    return {'wikicaps_id': wikicaps_id,
            'caption': caption}


def get_wikicaps_ids(top50_indices: List[int], dataset: WICSMMIRDataset):
    wids = []
    for idx in top50_indices:
        wids.append(get_wicsmmir_data(int(idx), dataset)['wikicaps_id'])
    return wids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wicsmmir_images', type=str,
                        default='/raid/datasets/wicsmmir/images')
    parser.add_argument('--ranks_file', type=str,
                        default=f'{os.getcwd()}/rankswicsmmirv2.pth.tar')
    parser.add_argument('--config', type=str,
                        default=f'{os.getcwd()}/configs/teran_wicsmmir_v2_MrSw.yaml',
                        help="Which configuration to use for to load the dataset")

    opts = parser.parse_args()
    assert os.path.lexists(opts.wicsmmir_images), "WICSMMIR Images Path does not exist!"
    assert os.path.lexists(opts.ranks_file), "Ranks file does not exist!"
    assert os.path.lexists(opts.config), "Config file does not exist!"

    # load config file
    print("loading config file")
    opt = parser.parse_args()
    if opt.config is not None:
        with open(opt.config, 'r') as ymlfile:
            config = yaml.load(ymlfile)

    # load WICSMMIR dataset
    print("loading WICSMMIR dataset")
    test_set_file = config['dataset']['test_set']
    features_root = config['dataset']['features_root']
    random_seed = config['dataset']['random_seed']
    dataset = WICSMMIRDataset(features_root, test_set_file, True, random_seed)

    # load top50 ranks from eval output
    print("loading top50 ranks from eval output")
    top50 = load_top50(opts)

    # get wicsmmir data
    print("getting wicsmmir data")
    data = []
    for cap_idx, t50 in enumerate(tqdm(top50)):
        cap_data = get_wicsmmir_data(cap_idx, dataset)
        top50_wids = get_wikicaps_ids(t50, dataset)
        cap_data['top50_wids'] = top50_wids
        data.append(cap_data)

    # create and persist result Dataframe
    df_outfile = "user_study_data.df.feather"
    print(f"persisting result DataFrame at {df_outfile}")
    res = pd.DataFrame(data)
    res.to_feather(df_outfile)

    # create archive containing all images
    archive_file = "wicsmmir_images.tar.gz"
    print(f"creating archive containing all images at {archive_file}")
    archive = tar.open(archive_file, "w:gz")


    def reset(tarinfo):
        tarinfo.uid = tarinfo.gid = 0
        tarinfo.uname = tarinfo.gname = "root"
        return tarinfo


    for wid in tqdm(res['wikicaps_id']):
        archive.add(os.path.join(opts.wicsmmir_images, f"wikicaps_{wid}.png"),
                    filter=reset)

    # add dataframe to archive file
    archive.add(df_outfile, filter=reset)
    archive.close()
