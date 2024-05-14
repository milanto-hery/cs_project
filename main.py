import json
from Preprocess_CS_2D import *
import os

def load_json_config(file_name):
    try:
        with open(file_name, 'r') as file:
            config = json.load(file)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{file_name}': {e}")
        return None

def main():
    # Set the species folder as working directory. The Audio, Annotaions and DataFiles folder should be inside this
    species_folder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(species_folder)
    # Load configuration from JSON
    config = load_json_config('settings.json')
    if config:
        print(config)

    species_name = 'thyolo_alethe' # just specify the name of your species here, the parameters already define in the json
    species_params = config[species_name]  # this for thyolo alethe

    segment_duration = species_params['segment_duration']
    positive_class = species_params['positive_class']
    negative_class = species_params['negative_class']
    file_type = species_params['file_type']
    audio_extension = species_params['audio_extension']
    n_fft = species_params['n_fft']
    hop_length = species_params['hop_length']
    n_mels = species_params['n_mels']
    f_min = species_params['f_min']
    f_max = species_params['f_max']
    solver = species_params['solver_lasso']

    # manually choose the batch and threads based on the cores available on the computer you used
    batch_size =10
    max_threads = 10
    R = 0.10 # specify the number of small samples
    saved_folder = './CS_2D_output/Saved_10' # according to the small samples, specify the name of folder to save the outputs

    pre_process = Preprocess_CS_2D(species_folder, segment_duration,
                 positive_class, negative_class,
                 n_fft, hop_length, n_mels, f_min, f_max,file_type, audio_extension, solver, R)
    pre_process.create_dataset(max_threads, batch_size, saved_folder, False)

if __name__ == "__main__":
    main()
