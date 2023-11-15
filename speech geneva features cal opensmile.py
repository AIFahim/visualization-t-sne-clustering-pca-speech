import opensmile
import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm

# Function to process a single audio file and return the DataFrame
def process_audio_file(audio_file_path):
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        df = smile.process_file(audio_file_path)
        
        # Extract file GUID/name from the audio file path
        file_guid_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        # Add the GUID/name as the first column in the DataFrame
        df.insert(0, 'GUID_name', file_guid_name)
        
        return df
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None

# Function to process multiple audio files using multiple workers
def process_audio_files_with_workers(audio_files_list, num_workers, output_csv_file):
    with Pool(num_workers) as pool, open(output_csv_file, 'w') as f_out:
        # Write CSV header
        f_out.write(','.join(process_audio_file(audio_files_list[0]).columns) + '\n')

        for df in tqdm(pool.imap(process_audio_file, audio_files_list), total=len(audio_files_list)):
            if df is not None:
                df.to_csv(f_out, header=False, index=False)

if __name__ == "__main__":
    audio_files_directory = "/home/asif/viz_eda_max_non_speech_regions/clusters_3_cutted_500ms/cluster_0/subcluster_3/cluster_0"
    output_csv_file = "clusters_3_cutted_500ms_cluster_0_subcluster_3_cluster_0.csv"
    num_workers = 4  # You can adjust this number based on your system's capabilities

    audio_files_list = [os.path.join(audio_files_directory, file) for file in os.listdir(audio_files_directory) if file.endswith(".flac")]

    process_audio_files_with_workers(audio_files_list, num_workers, output_csv_file)

    print("Real-time saving complete.")


# import opensmile

# smile = opensmile.Smile(
# feature_set=opensmile.FeatureSet.ComParE_2016,
# feature_level=opensmile.FeatureLevel.Functionals,
# )

# # the result is a pandas.DataFrame containing the features
# y = smile.process_file('/home/asif/vizualizations/noise_segments_data/max_non_speech_segments/0a0d23b6-4b50-4ccb-b0fb-ee5779d3ebd7_max_nonspeech.flac')

# print(y.columns)
