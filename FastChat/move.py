import os
import pickle

nina_root = 'output/vicuna/15_write'
anna_root = 'output/vicuna/15_write_anna'

for file in os.listdir(nina_root):
    if os.path.isdir(os.path.join(nina_root, file)):
        continue

    anna_correct_dir = os.path.join(anna_root, file[0], file[1])
    nina_correct_dir = os.path.join(nina_root, file[0], file[1])

    if os.path.exists(os.path.join(nina_correct_dir, file)):
        # print('!!!!!!! unlink', os.path.join(nina_root, file))
        os.unlink(os.path.join(nina_root, file))
    else:
        os.makedirs(anna_correct_dir, exist_ok=True)

        # print('link', os.path.join(nina_root, file), os.path.join(anna_correct_dir, file))
        os.link(os.path.join(nina_root, file), os.path.join(anna_correct_dir, file))

        # print('unlink', os.path.join(nina_root, file))
        os.unlink(os.path.join(nina_root, file))

