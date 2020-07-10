import numpy as np
import pandas as pd
from batchlib.util import image_name_to_well_name


def to_well_names(image_names):
    well_names = [image_name_to_well_name(im_name)
                  for im_name in image_names]
    return np.array(well_names), np.unique(well_names)


def run_well_analysis(input_table_name, output_table_name):
    table = pd.read_excel(input_table_name)

    image_names = table['image_name'].values
    well_names, unique_wells = to_well_names(image_names)

    sensor_intensities = table['sensor'].values
    sensor_thresholds = [7000, 8000]
    marker_intensities = table['infection marker'].values
    marker_thresholds = [120]

    def analyse_well(well_name):
        well_mask = well_names == well_name
        n_cells = well_mask.sum()

        this_sensor_intensities = sensor_intensities[well_mask]
        sensor_fractions = []
        for sensor_threshold in sensor_thresholds:
            sensor_fraction = (this_sensor_intensities > sensor_threshold).sum() / float(n_cells) * 100
            sensor_fractions.append(sensor_fraction)

        this_marker_intensities = marker_intensities[well_mask]
        marker_fractions = []
        for marker_threshold in marker_thresholds:
            marker_fraction = (this_marker_intensities > marker_threshold).sum() / float(n_cells) * 100
            marker_fractions.append(marker_fraction)

        well_stats = [n_cells] + sensor_fractions + marker_fractions
        return well_stats

    col_names = ['well_name', 'n_cells']
    col_names.extend([f'percent sensor intensity > {sensor_threshold}'
                      for sensor_threshold in sensor_thresholds])
    col_names.extend([f'percent marker intensity > {marker_threshold}'
                      for marker_threshold in marker_thresholds])

    out_table = []
    for well_name in unique_wells:
        well_stats = analyse_well(well_name)
        print(well_name)
        print(well_stats)
        print()
        out_table.append([well_name] + well_stats)

    out_table = np.array(out_table)
    out_table = pd.DataFrame(out_table, columns=col_names)
    out_table.to_excel(output_table_name)


if __name__ == '__main__':
    in_table = './DS_plate1_conf_20200702_112943_491_cells_table.xlsx'
    out_table = './DS_plate1_conf_20200702_112943_491_well_table.xlsx'
    run_well_analysis(in_table, out_table)
