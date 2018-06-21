import os
import pandas as pd
import numpy as np
from pydub import AudioSegment

dataset_storage = "/media/renzo/HDD"
source = "/media/renzo/Amsterdam on Canvas/fma_large"
destination = dataset_storage + "/fma_large_wav"

genre_list = ['Electronic', 'Rock', 'Pop', 'Hip-Hop', 'Lo-Fi', 'Jazz', 'Classical']
counter = 0

for genre in genre_list:
    if not os.path.exists(destination + "/" + genre):
        os.mkdir(destination + "/" + genre)

df = pd.read_csv(dataset_storage + "/tracks.csv", low_memory=False)
df = df[pd.notnull(df["track.7"])]
print(df.loc[df["track.7"] == "Experimental"])

# for innermap in os.listdir(source):
#     print(innermap)
#     for audio_file in os.listdir(source + "/" + innermap):
#         track_id = str(int(audio_file[:-4]))
#         genre = df.loc[df["Unnamed: 0"] == track_id]["track.7"].item()
#         if genre in genre_list:
#             counter += 1
#             if counter > 31208:
#                 try:
#                     sound = AudioSegment.from_mp3(source + "/" + innermap + "/" + audio_file)
#                     sound.export(destination + "/" + genre + "/" + genre + "_" + str(counter) + ".wav", format="wav")
#                 except:
#                     print("\nRemoving: " + innermap + "/" + audio_file)
#                     os.remove(source + "/" + innermap + "/" + audio_file)
#                     print("file is removed and the algorithm keeps on running...")
#                     counter -= 1