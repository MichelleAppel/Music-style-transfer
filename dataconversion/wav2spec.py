import os
import argparse
import librosa
import pickle
import numpy as np
import math

def fastfft(y):
	# STFT
	D = librosa.stft(y, n_fft=254)
	spectrogram, phase = librosa.magphase(D)
	spectrogram = np.log1p(spectrogram)
	return spectrogram

def wav2spec(wav, time_sec=30, sample_rate=22050):
	# Read audio
	wav = wav[:round(time_sec*sample_rate)]
	spectrogram = fastfft(wav)

	return spectrogram

def spec2pickle(spectrogram, filename):
	filename = filename[:-3]  # remove .au from filename
	np.save(filename, spectrogram)

def dir2specs(source, destination):
	if not os.path.exists(destination):
		os.mkdir(destination)
	for genre in os.listdir(source):
		if genre != '.DS_Store':  # for Mac (hidden file)
			for filename in os.listdir(source+'/'+genre):
				wav, sample_rate = librosa.core.load(source+'/'+genre+'/'+filename)
				spectrogram = wav2spec(wav)
				spectrogram = crop_data(spectrogram)
				if not os.path.exists(destination+'/'+genre):
					os.mkdir(destination+'/'+genre)
				spec2pickle(spectrogram, destination+'/'+genre+'/'+filename)

def crop_data(spectrogram):
	spectrogram = spectrogram[1:, :]
	width = np.shape(spectrogram)[1]
	print(2**math.floor(math.log(width, 2)))
	return spectrogram[:, :2**math.floor(math.log(width, 2))]


def dir2specs(source, destination, time_sec=30, sample_rate=22050, crop=True):
	if not os.path.exists(destination):
		os.mkdir(destination)
	for genre in os.listdir(source):
		if genre != '.DS_Store': # for Mac (hidden file)
			for filename in os.listdir(source+'/'+genre):
				wav, sample_rate = librosa.core.load(source+'/'+genre+'/'+filename)
				spectrogram = wav2spec(wav, time_sec, sample_rate)
				if crop:
					spectrogram = crop_data(spectrogram)
					print(np.shape(spectrogram))

				if not os.path.exists(destination+'/'+genre):
					os.mkdir(destination+'/'+genre)
				spec2pickle(spectrogram, destination+'/'+genre+'/'+filename)

def crop_data(spectrogram):
	width = np.shape(spectrogram)[1]
	return spectrogram[:, :2**math.floor(math.log(width, 2))]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source', default=None, required=False)
	parser.add_argument('-d', '--destination', default=None, required=False)
	parser.add_argument('-sr', '--sample_rate', default=22050, required=False)
	parser.add_argument('-ts', '--time_sec', default=0.4, required=False)
	parser.add_argument('-c', '--crop', default=True, required=False)
	args = vars(parser.parse_args())

	if args['source'] == None:
		args['source'] = '../datasets/genres_selection/'

	if args['destination'] == None:
		args['destination'] = './output/dummydata/sec='+str(args['time_sec'])+'_sr='+str(args['sample_rate'])+'_crop='+str(args['crop'])

	dir2specs(args['source'], args['destination'], float(args['time_sec']), int(args['sample_rate']), bool(args['crop']))

if __name__ == "__main__":
	main()
