import detect_chorus
import detect_f20p
import os

# survey_path = './data/raw_data/'
survey_path = 'C:/Users/cleap/Documents/Data/Sound Data/Miller'
save_path = './data/'

for location_folder_name in os.scandir(survey_path):
    location_folder_path = os.path.join(survey_path, location_folder_name)
    if os.path.isdir(location_folder_path):
        wav_folder_path = os.path.join(location_folder_path, 'wav')
        if os.path.exists(wav_folder_path):
            detect_chorus.plot_noise_distribution(survey_path=wav_folder_path, save_path=save_path)
            # detect_f20p.detect_f20p(survey_path=wav_folder_path, save_path=save_path)
            # detect_chorus.detect_chorus(wav_folder_path)
