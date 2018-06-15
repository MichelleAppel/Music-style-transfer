
import os
import argparse
import librosa
import pickle
import numpy as np
import scipy

SAMPLE_RATE = 21050
RESTORE_ITER = 50 # 50 iterations -> takes ~ 5-10s per file

def phase_restore(mag, random_phase, restore_iter):
	p = np.exp(1j * (random_phase))
	init_shape = np.shape(p)

	for i in range(restore_iter):
		_, p = librosa.magphase(librosa.stft(librosa.istft(mag * p), n_fft=254))

	return p

def spec2wav(spectrogram, restore_iter):
	random_phase = spectrogram.copy()
	np.random.shuffle(random_phase)
	p = phase_restore((np.exp(spectrogram) - 1), random_phase, restore_iter)
	reconstructed_wav = librosa.istft((np.exp(spectrogram) - 1) * p)
	return reconstructed_wav

def save_wav(spectrogram, filename, sample_rate):
	filename = filename[:-4] # remove .npy from filename
	filename += '_reconstr.wav'
	scipy.io.wavfile.write(filename, sample_rate, spectrogram)
	
def dir2wavs(source, destination, restore_iter):
	print(destination)
	if not os.path.exists(destination):
		os.mkdir(destination)
	for genre in os.listdir(source):
		if genre != '.DS_Store': # for Mac (hidden file)
			for filename in os.listdir(source+'/'+genre):
				spectrogram = np.load(source+'/'+genre+'/'+filename)
				reconstructed_wav = spec2wav(spectrogram, restore_iter=restore_iter)
				if not os.path.exists(destination+'/'+genre):
					os.mkdir(destination+'/'+genre)
				save_wav(reconstructed_wav, destination+'/'+genre+'/'+filename, sample_rate=SAMPLE_RATE)
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source', default=None, required=False)
	parser.add_argument('-d', '--destination', default=None, required=False)
	parser.add_argument('-sr', '--sample_rate', default=22050, required=False)
	parser.add_argument('-ri', '--restore_iter', default=50, required=False)
	args = vars(parser.parse_args())
	
	if args['source'] == None:
		args['source'] = './output/dummydata/sec=5_sr=22050_crop=True'

	pos = [i for i, ltr in enumerate(args['source']) if ltr == '/'][-1]
	if args['destination'] == None:
		args['destination'] = './output/reconstructed/' + args['source'][pos+1:]+'_ri='+str(args['restore_iter'])
	
	dir2wavs(args['source'], args['destination'], args['restore_iter'])
	
if __name__ == "__main__":
	main()

