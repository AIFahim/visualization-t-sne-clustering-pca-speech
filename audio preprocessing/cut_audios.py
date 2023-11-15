import os
from pydub import AudioSegment

def cut_audio(input_file, output_file, cut_time_ms=100):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Calculate the number of milliseconds to cut from each end
    cut_duration = cut_time_ms
    if cut_time_ms > len(audio):
        raise ValueError("Cut time is longer than the audio duration.")

    # Trim the audio by cutting the specified amount of milliseconds from the front and back
    trimmed_audio = audio[cut_duration:-cut_duration]

    # Save the trimmed audio to the output file
    trimmed_audio.export(output_file, format="flac")

def process_audio_files(folder_path, output_folder, cut_time_in_milliseconds=100):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of audio files in the folder
    audio_files = [file for file in os.listdir(folder_path) if file.endswith(".flac")]

    # Process each audio file
    for audio_file in audio_files:
        input_file = os.path.join(folder_path, audio_file)
        output_file = os.path.join(output_folder, audio_file)

        try:
            cut_audio(input_file, output_file, cut_time_in_milliseconds)
            print(f"Audio '{audio_file}' cut successfully.")
        except Exception as e:
            print(f"Error while cutting audio '{audio_file}':", e)

if __name__ == "__main__":
    input_folder = "max_non_speech_segments"
    output_folder = "cutted_max_non_speech_segments"

    # Specify the cut time in milliseconds (100ms in this case)
    cut_time_in_milliseconds = 100

    try:
        process_audio_files(input_folder, output_folder, cut_time_in_milliseconds)
        print("All audio files processed successfully.")
    except Exception as e:
        print("Error while processing audio files:", e)
