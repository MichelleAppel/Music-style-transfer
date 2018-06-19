import os
import pandas as pd
from pydub import AudioSegment

dataset_storage = "/media/renzo/HDD"
source = dataset_storage + "/fma_small"
destination = dataset_storage + "/fma_large_wav"

df = pd.read_csv(dataset_storage + "/fma_metadata/tracks.csv", low_memory=False)
counter = 0

for innermap in os.listdir(source):
    for audio_file in os.listdir(source + "/" + innermap):
        track_id = str(int(audio_file[:-4]))
        genre = df.loc[df["Unnamed: 0"] == track_id]["track.7"].item()
        if innermap == "001" and track_id == "1083":
            print(genre)
        if genre in ['Electronic', 'Rock', 'Pop', 'Hip-Hop', 'Lo-Fi', 'Jazz', 'Classical', 'Experimental']:
            if not os.path.exists(destination + "/" + genre):
                os.mkdir(destination + "/" + genre)
            try:
                sound = AudioSegment.from_mp3(source + "/" + innermap + "/" + audio_file)
                sound.export(destination + "/" + genre + "/" + genre + "_" + str(counter) + ".wav", format="wav")
                counter += 1
            except:
                print("\nPress Enter to remove: " + innermap + "/" + audio_file)
                input("Press Ctrl+C to quit!")
                os.remove(source + "/" + innermap + "/" + audio_file)
                print("file is removed and the algorithm keeps on running...")