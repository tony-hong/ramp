import pickle


path = f'data/howto100m_sentencified/sentencified_htm_1200k_with_audio.pickle'
exclude_path = f'data/howto100m_sentencified/sentencified_htm_10k_with_audio.pickle'
output_path = f'output/vicuna/1200k_wo_10k.pickle'

with open(path, 'rb') as fin:
    data = pickle.load(fin)

with open(exclude_path, 'rb') as fin:
    exclude_data = pickle.load(fin)

for key in exclude_data:
    del data[key]

print("New size", len(data))


with open(output_path, 'wb') as fout:
    pickle.dump(data, fout)
