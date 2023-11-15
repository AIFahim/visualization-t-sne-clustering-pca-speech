import os
from pydub import AudioSegment
from multiprocessing import Pool
from tqdm import tqdm

def cut_audio(input_file, output_file, cut_time_ms=100):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Calculate the number of milliseconds to cut from each end
    cut_duration = cut_time_ms
    if cut_time_ms > len(audio):
        raise ValueError("Cut time is longer than the audio duration.")

    # Trim the audio by cutting the specified amount of milliseconds from the front and back
    trimmed_audio = audio[cut_duration:-cut_duration]

    # # Save the trimmed audio to the output file
    # trimmed_audio.export(output_file, format="flac")

    # Check if the duration of trimmed audio is greater than 1 second (1000 milliseconds)
    if len(trimmed_audio) >= 1000:
        # Save the trimmed audio to the output file
        trimmed_audio.export(output_file, format="flac")

def process_audio_file(args):
    input_file, output_file, cut_time_in_milliseconds = args
    try:
        cut_audio(input_file, output_file, cut_time_in_milliseconds)
        return f"Audio '{input_file}' cut successfully."
    except Exception as e:
        return f"Error while cutting audio '{input_file}': {e}"

def process_audio_files(input_folder, output_folder, cut_time_in_milliseconds=100):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of audio files in the folder
    audio_files = [file for file in os.listdir(input_folder) if file.endswith(".flac")]

    # Prepare arguments for multiprocessing
    args_list = []
    for audio_file in audio_files:
        input_file = os.path.join(input_folder, audio_file)
        output_file = os.path.join(output_folder, audio_file)
        args_list.append((input_file, output_file, cut_time_in_milliseconds))

    # Use tqdm to create a progress bar
    with Pool() as pool, tqdm(total=len(args_list), desc="Processing audio files") as pbar:
        for result in pool.imap_unordered(process_audio_file, args_list):
            print(result)
            pbar.update(1)

if __name__ == "__main__":
    input_folder = "max_non_speech_segments"
    output_folder = "cutted_max_non_speech_segments_multipro"

    # Specify the cut time in milliseconds (100ms in this case)
    cut_time_in_milliseconds = 500

    try:
        process_audio_files(input_folder, output_folder, cut_time_in_milliseconds)
        print("All audio files processed successfully.")
    except Exception as e:
        print("Error while processing audio files:", e)
