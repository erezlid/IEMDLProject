from datasets import load_dataset
import requests
import os
from tqdm import tqdm

ds_name = 'yerevann/coco-karpathy'
folder = '/home/erezlid/CapDec/data/images_coco/val2014'


def download_images(data_split):
    for i in tqdm(range(len(data_split)), desc = 'Downloading Images'):
        url = data_split[i]['url']
        response = requests.get(url)
        filename = url.split('/')[-1]
        if response.status_code != 200:
            print(f"Error downloading {filename}")
            continue
        with open(os.path.join(folder, filename), 'wb') as f:
            f.write(response.content)


def main():
    val = load_dataset(ds_name, split='validation')
    download_images(val)
    print("Finished Downloading")


if __name__ == "__main__":
    main()

