import os
from glob import glob


def check_plates():
    root = '/g/kreshuk/data/covid/data-processed'
    plates = glob(os.path.join(root, '*DS*'))

    for plate in plates:
        name = os.path.split(plate)[1]
        table_path = os.path.join(plate, f'{name}_cells_table.xlsx')

        if os.path.exists(table_path):
            print("Plate", name, "complete")
        else:
            print("Plate", name, "INCOMPLETE")


if __name__ == '__main__':
    check_plates()
