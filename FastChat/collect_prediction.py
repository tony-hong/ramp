import os.path
import pickle
import tqdm
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str,)
parser.add_argument("--words", type=str, default='200')
parser.add_argument("--subset", type=str, default='10k')
parser.add_argument("--whisper_asr", type=str, default=None)
args = parser.parse_args()

words = args.words
exp_name = args.exp


if args.whisper_asr is None:
    path = f'data/howto100m_sentencified/sentencified_htm_{args.subset}_with_audio.pickle'
else:
    path = f'output/whisper/{args.whisper_asr}_{args.subset}_postprocessed.pickle'

with open(path, 'rb') as fin:
    data = pickle.load(fin)

preload_captions = None
PRELOAD = ['200k', '100k', '10k']
for subset in PRELOAD:
    path = f'output/vicuna/{exp_name}_{subset}.pickle'
    if os.path.exists(path):
        print(f"Using pre-collected captions from: {path}")
        with open(path, 'rb') as fin:
            preload_captions = pickle.load(fin)
        break


for id_, (key, val) in enumerate(tqdm.tqdm(data.items())):
    if preload_captions is not None and key in preload_captions:
        # print("Using pre-collected", flush=True)
        val[f'{words}w']['prediction'] = preload_captions[key][f'{words}w']['prediction']
        # print(preload_captions[key][f'{words}w']['prediction'], flush=True)
        continue


    nina_result_dir = os.path.join('output/vicuna/', exp_name)
    anna_result_dir = os.path.join('output/vicuna/', exp_name + '_anna')

    old_output_path = f'{nina_result_dir}/{key}.pickle'
    nina_output_path = f'{nina_result_dir}/{key[0]}/{key[1]}/{key}.pickle'
    anna_output_path = f'{anna_result_dir}/{key[0]}/{key[1]}/{key}.pickle'

    if os.path.exists(nina_output_path):
        path = nina_output_path
    elif os.path.exists(anna_output_path):
        path = anna_output_path
    elif os.path.exists(old_output_path):
        path = old_output_path
    else:
        print(f'OUTPUT for {key} does not exist')
        raise ValueError

    try:
        with open(path, 'rb') as fin:
            val[f'{words}w']['prediction'] = pickle.load(fin)
    except Exception as e:
        print(e, file=sys.stderr)
        val[f'{words}w']['prediction'] = ['' * len(val[f'{words}w']['text'])]

output_path = os.path.join('output/vicuna/', f'{exp_name}_{args.subset}')

with open(output_path + '.pickle', 'wb') as fout:
    pickle.dump(data, fout)



