import os
import argparse
import librosa
import pickle
import numpy as np
import math
import scipy.misc
import random
from tqdm import tqdm

def fastfft(y):
	# STFT
	D = librosa.stft(y, n_fft=2046)
	spectrogram, phase = librosa.magphase(D)
	spectrogram = np.log1p(spectrogram)
	return spectrogram

# Add Gaussian noise to image
def add_noise(wav, mean=0, std_max_factor=230):
	#TODO: check best value for std_max_factor (200 results in too much noise, 250 too little)
	
	# Set the std as a fraction of the max value in the wav
	# (for loud wavs, the noise will be higher)
	std = np.max(wav) / std_max_factor
	
	# Generate Gaussian noise and add to wav
	noise = np.random.normal(mean, std, np.shape(wav))

	return wav+noise

def wav2wavs(source, filename, genre, output_path, time_sec, time_steps, max_time, over_write_file):
	for i in range(0, max_time, time_steps):
		
		# Cut audio from i to i+time_sec and add noise
		wav, sample_rate = librosa.load(source+'/'+genre+'/'+filename, offset=float(i), duration=time_sec)
		wav_noise = add_noise(wav)
		
		# Determine file name
		new_filename = filename[:-3].replace('.', '_') #TODO make the -3 (for .au) more robust for other extensions
		new_filename += '_'+str(i).zfill(2)+'-'+str(int(i+time_sec)).zfill(2)
		
		# Transform to spectrogram using Short-Time Fourier Transform
		# Furthermore, divide by max to scale max value to 1
		spectrogram = fastfft(wav)
		spectrogram = spectrogram / np.max(spectrogram)
		spectrogram_noise = fastfft(wav_noise)
		spectrogram_noise = spectrogram_noise / np.max(spectrogram_noise)	
		
		# If file does not exist or over_write_file == True, save wav, jpg, and npy		
		if not os.path.isfile(output_path+'/'+genre+'_wav/'+new_filename+'_augm.wav') or over_write_file:
			
			# save spectrograms as numpy matrices
			np.save(output_path+'/'+genre+'_npy/'+new_filename+'_augm.npy', spectrogram_noise) # with noise
			#np.save(output_path+'/'+genre+'_npy/'+new_filename+'_norm.npy', spectrogram) # without noise
			
			# save wavs
			librosa.output.write_wav(output_path+'/'+genre+'_wav/'+new_filename+'_augm.wav', wav_noise, sample_rate) # with noise
			#librosa.output.write_wav(output_path+'/'+genre+'_wav/'+new_filename+'_norm.wav', wav, sample_rate) # without noise
			
			# save spectrograms as jpgs
			scipy.misc.imsave(output_path+'/'+genre+'_jpg/'+new_filename+'_augm.jpg', spectrogram_noise) # with noise
			#scipy.misc.imsave(output_path+'/'+genre+'_jpg/'+new_filename+'_norm.jpg', spectrogram) # without noise
			
		else:
			print('Not saving files. They already exist.')
			break
		
def dir2specs(source, output_path, time_sec, time_steps, max_time, over_write_file):
	# Create the output directory if not yet existing
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	
	# Get all genre names (without hidden files & sorted)
	genres = sorted([g for g in os.listdir(source) if not g.startswith('.')])
	for genre in genres:
		print(genre)
		# Get all file names (without hidden files & sorted)
		file_names = sorted([f for f in os.listdir(source+'/'+genre) if not f.startswith('.')])
		for file_name in tqdm(file_names): # tqdm for progress bar
			wav2wavs(source, file_name, genre, output_path, time_sec, time_steps, max_time, over_write_file)
			
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source', default='two_genres', required=False)
	parser.add_argument('-d', '--destination', default='new_augmented_data', required=False)
	parser.add_argument('-tsec', '--time_sec', default=5.0, required=False)
	parser.add_argument('-tstep', '--time_steps', default=3, required=False)
	parser.add_argument('-maxt', '--max_time', default=25, required=False)
	parser.add_argument('-owf', '--over_write_file', default=True, required=False)
	parser.add_argument('-sn', '--save_normal', default=False, required=False)	
	
	args = vars(parser.parse_args())

	dir2specs(args['source'], args['destination'], float(args['time_sec']), int(args['time_steps']), 
		int(args['max_time']), bool(args['over_write_file']), bool(args['save_normal']))
		
if __name__ == "__main__":
	main()

