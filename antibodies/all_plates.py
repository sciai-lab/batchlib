from glob import glob
from instance_analysis_workflow2 import run_instance_analysis2, parse_instance_config2
# from run_all import run_all


# TODO implement run_all from config
# def all_plates_all_wfs():
#     in_folder = '/home/covid19/data/covid-data-vibor'
#     folders = glob(in_folder + '/*')
# 
#     for folder in folders:
#         print("Run for plate", folder)
#         run_all(folder, 0, 12)


def load_config(folder, key):
    config = parse_instance_config2()
    config.input_folder = folder
    config.use_unique_output_folder = True
    config.in_analysis_key = key
    return config


def all_plates_instance_wf2(with_corrected=True):
    in_folder = '/home/covid19/data/covid-data-vibor'
    folders = glob(in_folder + '/*')

    for folder in folders:
        
        config = load_config(folder, 'raw')
        print("Run instance analysis 2 for plate", config.input_folder)
        print("with raw data")
        run_instance_analysis2(config)

        if with_corrected:
            config = load_config(folder, 'corrected')
            print("Run instance analysis 2 for plate", config.input_folder)
            print("run with corrected data")
            run_instance_analysis2(config)


# all_plates()
all_plates_instance_wf2()
