import pandas as pd
import os


def make_image_list(initial_table, root):
    initial_table = pd.read_excel(initial_table)

    plate_names = initial_table['plate'].values
    file_names = initial_table['file'].values
    status = initial_table['infected'].values

    paths = []
    for plate_name, name, stat in zip(plate_names, file_names, status):
        if stat == 1.:
            continue
        plate_name = plate_name.replace('_IgG', '').replace('_IgA', '')
        path = os.path.join(root, plate_name, name + '.h5')
        paths.append(path)

    paths = list(set(paths))
    paths.sort()

    for p in paths:
        print(p)


if __name__ == '__main__':
    initial_table = './Stacks2proofread.xlsx'
    root = ''
    make_image_list(initial_table, root)
