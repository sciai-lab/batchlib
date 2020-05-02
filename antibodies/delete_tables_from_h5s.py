import os
from glob import glob
from tqdm.auto import tqdm
from batchlib.util import open_file


def delete_key_from_h5s_in_dir(directory, keys=('tables')):
    i = 0
    for file in tqdm(glob(os.path.join(directory, '*.h5')), desc=f'deleting keys {keys} from h5s in {directory}'):
        with open_file(file, 'a') as f:
            for key in keys:
                if key in f.keys():
                    i += 1
                    del f[key]
    print(f'deteted {i} datasets/groups')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dirs', nargs='+', type=str, help='directories to delte tables in h5s from')
    parser.add_argument('-k', '--keys', nargs='+', default=['tables'], type=str, help='keys to delete')
    args = parser.parse_args()
    for d in args.data_dirs:
        if d.endswith(os.path.sep):
            d = d[:-1]
        delete_key_from_h5s_in_dir(d, args.keys)
