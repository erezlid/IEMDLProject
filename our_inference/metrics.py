import argparse
import json
import numpy as np
import os

from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from constant_config import VALIDATION_FILE_PATH, VALIDATION_EMBEDDING_PATH,RESULTS_FOLDER,MODEL_NAME

model = SentenceTransformer(MODEL_NAME)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def sort_json_data(json_data):
    return sorted(json_data, key=lambda k: k.get('image_id'))

def get_sorted_data(generated_path, val_path):
    sorted_val_data = None
    generated_data = load_json(generated_path)
    sorted_generated_data = sort_json_data(generated_data)
    if not os.path.exists(VALIDATION_EMBEDDING_PATH):
        val_data = load_json(val_path)
        sorted_val_data = sort_json_data(val_data)
    return sorted_generated_data, sorted_val_data

def get_generated_caption_embeddings(sorted_generated_data):
    generated_embeddings = {}
    for generated_caption in tqdm(sorted_generated_data, desc="Generating Embeddings"):
        if generated_caption['image_id'] not in generated_embeddings:
            generated_embeddings[generated_caption['image_id']] = []
        generated_embeddings[generated_caption['image_id']].append(model.encode(generated_caption['caption']))
    return generated_embeddings

def get_val_caption_embeddings(val_data):

    if os.path.exists(VALIDATION_EMBEDDING_PATH):
        print("Loading val embeddings")
        return load_json(VALIDATION_EMBEDDING_PATH)

    val_embeddings = {}
    for val_caption in tqdm(val_data, desc="Generating True Embeddings"):
        if val_caption['image_id'] not in val_embeddings:
            val_embeddings[val_caption['image_id']] = []

        embbeds = model.encode(val_caption['caption'])
        val_embeddings[val_caption['image_id']].append(embbeds.tolist())

    write_json_file(VALIDATION_EMBEDDING_PATH, val_embeddings)
    print("Saved Val Embeddings")

    return load_json(VALIDATION_EMBEDDING_PATH)



def compute_similarities(val_img_id, val_embedding, generated_embeddings):
    outputs = []
    for generated_embedding in generated_embeddings[val_img_id]:
        sim =  1 - distance.cosine(val_embedding, generated_embedding)
        outputs.append(sim)
    return np.mean(outputs)


def main_loop(generated_embeddings, val_embeddings_dct):
    results = {}

    for val_img_id, val_embeddings in tqdm(val_embeddings_dct.items(), desc='Calculating Similarity'):
        similarity_outputs = []
        val_img_id = int(val_img_id)

        if val_img_id in generated_embeddings:
            for val_embedding in val_embeddings:
                val_embedding = np.array(val_embedding)
                similarity = compute_similarities(val_img_id, val_embedding, generated_embeddings)
                similarity_outputs.append(similarity)

            results[val_img_id] = np.mean(similarity_outputs)

    return results

def write_json_file(output_path,results):
    with open(output_path, 'w') as f:
        json.dump(results, f)


def main():

    parser = argparse.ArgumentParser(description="A simple argument parser example.")
    parser.add_argument('--generated_caption_file', type=str, default='')
    parser.add_argument('--output_file_name', type=str, default='')

    args = parser.parse_args()

    sorted_generated_data, sorted_val_data = get_sorted_data(args.generated_caption_file, VALIDATION_FILE_PATH)
    generated_embeddings = get_generated_caption_embeddings(sorted_generated_data)
    val_embeddings = get_val_caption_embeddings(sorted_val_data)
    results = main_loop(generated_embeddings, val_embeddings)
    write_json_file(os.path.join(RESULTS_FOLDER,args.output_file_name), results)

if __name__ == '__main__':
    main()










