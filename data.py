import hashlib
import math
import os
import pathlib
import tempfile
from typing import Generator
import zipfile
import sys

import click
import joblib
import numpy as np
import pandas as pd
import tqdm

from PIL import Image

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

def get_zip_file_list(zip_path: str) -> Generator[str, None, None]:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        yield from file_list

def get_files_df(enron_root: str) -> pd.DataFrame:
    all_files = []
    source_archives = []

    for zip_path in tqdm.tqdm(os.listdir(enron_root)):
        file_type = pathlib.Path(zip_path).suffix
        if file_type != '.zip':
            print('skipping', zip_path)
            continue
        zip_full_path = os.path.join(enron_root, zip_path)
        files_in_archive = list(get_zip_file_list(zip_full_path))
        all_files.extend(files_in_archive)
        source_archives.extend([zip_full_path] * len(files_in_archive))

    df = pd.DataFrame({
        'file_name': all_files,
        'source_archive': source_archives
    })
    df['filetype'] = df['file_name'].parallel_apply(lambda x: pathlib.Path(x).suffix)

    return df

valid_extensions = [
    '.jpg',
    '.jpeg',
    '.gif',
    '.png',
    '.mp3',
    '.wav',
    '.mpg',
    '.mpeg',
    '.mov',
    '.tif',
    '.tiff',
    '.bmp',
    '.avi',
    '.asf',
    '.wmv',
]

def extract_media(files_df: pd.DataFrame, media_dir: str):
    media_df = files_df.copy()
    media_df = media_df[media_df['filetype'].str.lower().isin(valid_extensions)]
    seen_files = set()
    with tempfile.TemporaryDirectory() as tmp_media_dir, tqdm.tqdm(total=len(media_df)) as pbar:
        for archive_path, files in media_df.groupby('source_archive'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for _, file in files.iterrows():
                    file_contents = zip_ref.read(file['file_name'])
                    file_name_without_path = os.path.basename(file['file_name'])
                    with open(os.path.join(tmp_media_dir, file_name_without_path), 'wb') as f:
                        f.write(file_contents)
                    if file_name_without_path in seen_files:
                        raise Exception(f'duplicate file {file_name_without_path}')
                    seen_files.add(file_name_without_path)
                    pbar.update(1)
        media_df['output_path'] = media_df['file_name'].apply(lambda x: os.path.join(tmp_media_dir, os.path.basename(x)))
        media_df['md5'] = media_df['output_path'].parallel_apply(lambda x: hashlib.md5(open(x, 'rb').read()).hexdigest())

        deduped_media_df = media_df.drop_duplicates('md5')
        os.makedirs(media_dir, exist_ok=True)
        for _, file in deduped_media_df.iterrows():
            os.rename(file['output_path'], os.path.join(media_dir, os.path.basename(file['file_name'])))

        deduped_media_df['final_path'] = deduped_media_df['file_name'].apply(lambda x: os.path.join(media_dir, os.path.basename(x)))

    return deduped_media_df

def try_open_image(path: str) -> tuple[int, int] | None:
    try:
        return Image.open(path).convert('RGB').size
    except:
        return None

def get_valid_images(deduped_media_files: pd.DataFrame):
    deduped_media_images = deduped_media_files.copy()
    deduped_media_images['img_shape'] = deduped_media_images['final_path'].parallel_apply(try_open_image)
    deduped_media_images = deduped_media_images[deduped_media_images['img_shape'].notna()]
    deduped_media_images['smallest'] = deduped_media_images['img_shape'].apply(lambda x: min(x))
    deduped_media_images = deduped_media_images[deduped_media_images['smallest'] > 1]
    return deduped_media_images


def get_embeddings_for_paths(image_paths: list[str], device: str = 'mps') -> torch.Tensor:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    inputs = processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outs = model.vision_model(**inputs)
    return vision_outs.pooler_output

def get_embeddings(deduped_media_images: pd.DataFrame, chunk_size: int = 50, device: str = 'mps') -> list[np.array]:
    embeddings = []
    for chunk_df in tqdm.tqdm(np.array_split(deduped_media_images, math.ceil(len(deduped_media_images) / chunk_size))):
        embeddings.extend(get_embeddings_for_paths(chunk_df['final_path'].tolist(), device=device).cpu().numpy())
    return embeddings

@click.command()
@click.option('--enron-root', type=click.Path(file_okay=False, exists=True), required=True)
@click.option('--media-dir', type=click.Path(), required=True)
@click.option('--output-path', type=click.Path(), required=True)
def dump_images_with_embeddings(enron_root: str, media_dir: str, output_path: str):
    print('Listing files...')
    files_df = get_files_df(enron_root)

    print()
    print('Extracting media...')
    media_df = extract_media(files_df, media_dir)

    print()
    print('Filtering for images...')
    images_df =  get_valid_images(media_df)

    print()
    print('Generating embeddings...')
    embeddings = get_embeddings(images_df)
    images_df['embedding'] = embeddings

    print()
    print('Writing to disk...')
    joblib.dump(images_df, output_path)

    print()
    print(f'df with embeddings saved to {output_path}')

 
if __name__ == '__main__':
    dump_images_with_embeddings()