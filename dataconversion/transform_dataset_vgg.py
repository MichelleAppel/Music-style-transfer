import os
import argparse
import numpy as np
from PIL import Image
import scipy.misc
from tqdm import tqdm

'''
This method loops through a directory with subdirectories (genres) containing samples in .npy format.
It transforms them to .jpg (with three channels), adapts the file_name so that it contains the genre, 
and puts them in one directory.
This will be used as dataset by the (music-genre-classification) deep learning models.
'''
def transform_dataset(source, destination, color):
	if not os.path.exists(destination):
		os.mkdir(destination)
		
	file_counter = 0
		
	# retrieve all genre directory names (without hidden files)
	all_genres = [f for f in os.listdir(source) if not f.startswith('.')]
	for genre in tqdm(all_genres):
				
		# retrieve all file names (without hidden files)
		all_files = [f for f in os.listdir(source+'/'+genre) if not f.startswith('.')]

		for filename in all_files:
			spectrogram = np.load(source+'/'+genre+'/'+filename)
			new_filename = destination + '/' + str(file_counter).zfill(4) + '_' + genre.lower() + '.jpg'
			
			if color == 'gray':
				scipy.misc.imsave(new_filename, spectrogram)
			else:
				rgb_spectrogram = np.zeros((1024, 256, 3), dtype=np.uint8)
				rgb_spectrogram[..., 0] = spectrogram * 255 / np.max(spectrogram)
				rgb_spectrogram[..., 1] = spectrogram * 255 / np.max(spectrogram)
				rgb_spectrogram[..., 2] = spectrogram * 255 / np.max(spectrogram)
				scipy.misc.imsave(new_filename, rgb_spectrogram)
			
			file_counter += 1
								
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source', default=None, required=True)
	parser.add_argument('-d', '--destination', default=None, required=True)
	parser.add_argument('-c', '--color', default='gray', required=False)
	
	args = vars(parser.parse_args())
	transform_dataset(args['source'], args['destination'], args['color'])

if __name__ == "__main__":
	main()
