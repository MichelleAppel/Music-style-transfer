import os
import argparse
import librosa
import pickle
import numpy as np
import math

def fastfft(y):
	# STFT
	D = librosa.stft(y, n_fft=2046)
	spectrogram, phase = librosa.magphase(D)
	spectrogram = np.log1p(spectrogram)
	return spectrogram

def wav2spec(wav, time_sec=10, sample_rate=22050, num_intervals=1):
	if num_intervals == 1: # Choose first x seconds
		wav = wav[:round(time_sec*sample_rate)]
		spectrogram = fastfft(wav)
		return [spectrogram]
	else: # Choose amount of intervals
		rest = np.shape(wav)[0] % num_intervals
		wav = wav[:-rest]
		wavs = np.array_split(wav, num_intervals)
		spectrograms = []
		for wav in wavs:
			spectrograms.append(fastfft(wav))
		return spectrograms

def spec2pickle(spectrogram, filename):
	filename = filename[:-3]  # remove .au from filename
	np.save(filename, spectrogram)

def dir2specs(source, destination, time_sec=10, sample_rate=22050, crop=True, num_intervals=1):
	if not os.path.exists(destination):
		os.mkdir(destination)
	for genre in os.listdir(source):
		if genre != '.DS_Store': # for Mac (hidden file)
			for filename in os.listdir(source+'/'+genre):
				wav, sample_rate = librosa.core.load(source+'/'+genre+'/'+filename)
				spectrograms = wav2spec(wav, time_sec, sample_rate, num_intervals)
				cropped_spectrograms = []
				if crop:
					for spectrogram in spectrograms:
						# print(spectrogram.shape)
						cr_spectrogram = crop_data(spectrogram)
						# print(cr_spectrogram.shape)
						cropped_spectrograms.append(cr_spectrogram)

				if not os.path.exists(destination+'/'+genre):
					os.mkdir(destination+'/'+genre)
				counter = 0
				for spectrogram in cropped_spectrograms:
					# print(str(int(counter)))
					spec2pickle(spectrogram, destination+'/'+genre+'/'+filename[:-3]+'_interval'+str(int(counter))+'npy')
					counter += 1

def crop_data(spectrogram):
	width = np.shape(spectrogram)[1]
	rest = width % 4
	return spectrogram[:, :width-rest]
	#return spectrogram[:, :2**math.floor(math.log(width, 2))]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source', default=None, required=False)
	parser.add_argument('-d', '--destination', default=None, required=False)
	parser.add_argument('-sr', '--sample_rate', default=22050, required=False)
	parser.add_argument('-ts', '--time_sec', default=10, required=False)
	parser.add_argument('-c', '--crop', default=True, required=False)
	parser.add_argument('-ni', '--num_intervals', default=1, required=False)
	args = vars(parser.parse_args())

	if args['source'] == None:
		args['source'] = '../datasets/genres_selection/'

	if args['destination'] == None:
		args['destination'] = './output/dummydata/sec='+str(args['time_sec'])+'_sr='+str(args['sample_rate'])+'_crop='+str(args['crop'])

	dir2specs(args['source'], args['destination'], float(args['time_sec']), int(args['sample_rate']), bool(args['crop']), int(args['num_intervals']))

if __name__ == "__main__":
	main()
