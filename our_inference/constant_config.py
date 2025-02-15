from enum import Enum


class ModelNames(Enum):
    ProxyLoss = 'ProxyLoss:'
    GradientNoise = 'GradientNoise:'
    NormInverse = 'NormInverse'
    N_Layers = 'N_Layers:'
    N_Heads = 'N_Heads:'
    T_Noise = 'T_Noise:'
    NoiseBySim = 'NoiseBySim:'

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

VALIDATION_FILE_PATH = './val_filtered.json'
VALIDATION_EMBEDDING_PATH = './val_embeddings.json'

RESULTS_FOLDER = './Results'
CAPDEC_RESULTS_FILE = 'capdec_005_results.json'
CAPTION_OUTPUT_FOLDER = './caption_outputs'
FIGS_FOLDER = './Figures'




