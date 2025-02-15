import argparse
import matplotlib.pyplot as plt
import os

from metrics import load_json
from constant_config import RESULTS_FOLDER,CAPDEC_RESULTS_FILE,FIGS_FOLDER,ModelNames

def get_model_hyperparameter_config(model_result_path):
    return "_".join(model_result_path.split('_')[:2]).capitalize()

def create_full_model_name(model_config):
    hyperparameter = model_config.split('_')[0].lower()
    if 'alpha' == hyperparameter:
        model_name = ModelNames.ProxyLoss.value + model_config
    elif 'stepsize' == hyperparameter:
        model_name = ModelNames.GradientNoise.value + model_config
    elif 'df' == hyperparameter:
        model_name = ModelNames.T_Noise.value + model_config
    elif 'layers' ==  hyperparameter:
        model_name = ModelNames.N_Layers.value + model_config
    elif 'heads' == hyperparameter:
        model_name = ModelNames.N_Heads.value + model_config
    elif "beta" or "minsim" or "samples" == hyperparameter:
        model_name = ModelNames.NoiseBySim.value + model_config
    else:
       model_name = ModelNames.NormInverse.value
    return model_name

def get_models_results(our_results_file):
    our_results_path = os.path.join(RESULTS_FOLDER,our_results_file)
    capdec_results_path = os.path.join(RESULTS_FOLDER,CAPDEC_RESULTS_FILE)
    return load_json(our_results_path), load_json(capdec_results_path)

def get_top15_images(capdec_results):

    results_sorted = sorted(capdec_results.items(), key=lambda x: x[1], reverse=True)
    top15_images = [item[0] for item in results_sorted[:15]]
    top15_similarities = [item[1] for item in results_sorted[:15]]
    return top15_images, top15_similarities

def filter_results_similarities(our_results,top15_images):
    filtered_results_similarities = [item[1] for item in our_results.items() if item[0] in top15_images]
    return filtered_results_similarities


def plot_results(top15_images,
                 top15_similarities,
                 filtered_results,
                 full_model_name):

    title = f'{full_model_name} VS CapDec Similarity Scores'
    plt.figure(figsize=(12,6))
    x_ticks = range(len(top15_images))

    plt.bar(x_ticks, top15_similarities, width=0.2, label='CapDec', align='center')
    plt.bar([i + 0.4 for i in x_ticks], filtered_results, width=0.2, label=full_model_name, align='center')

    plt.xticks(x_ticks, top15_images, rotation=45)
    plt.xlabel("Image ID")
    plt.ylabel("Similarity Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_FOLDER,
                             full_model_name+'.png'),
                dpi=500)

def main():
    parser = argparse.ArgumentParser(description="Plot Parser")
    parser.add_argument('--result_file', type=str)

    args = parser.parse_args()
    print("Getting Model Hyperparameter")
    model_hyperparameter_config = get_model_hyperparameter_config(args.result_file)
    print("Creating Model Full Name")
    model_full_name = create_full_model_name(model_hyperparameter_config)
    print("Getting Models Results")
    our_results, capdec_results = get_models_results(args.result_file)
    print("Getting Top15 Images")
    top15_images, top15_similarities = get_top15_images(capdec_results)
    print("Filtering CapDec Similarities")
    filtered_results_similarities = filter_results_similarities(our_results, top15_images)
    print("Plotting Results")
    plot_results(top15_images,
                 top15_similarities,
                 filtered_results_similarities,
                 model_full_name)


if __name__ == '__main__':
    main()










