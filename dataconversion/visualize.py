import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def save_plot(source_path):
    pos_left = [i for i, ltr in enumerate(source_path) if ltr == '/'][-1]
    pos_right = [i for i, ltr in enumerate(source_path) if ltr == '.'][-1]
    name = source_path[pos_left+1:pos_right]

    sr = 22050
    S = np.load(source_path) # Possibility to save result


    # Make a new figure
    plt.figure(figsize=(12,4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('Fourier spectrogram ' + name)

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()

    save_path = source_path+'_figure'

    plt.savefig(save_path + '.pdf')
    plt.savefig(save_path + '.png')

save_plot(source_path = '~/Documents/trained/augmented/results/model_100000_dataset_augmented_data/0/jazz_jazz_00004_18-23_augm.npy_original.npy')