import argparse
from pixel_analysis_workflow1 import run_pixel_analysis
from instance_analysis_workflow1 import run_instance_analysis1
from instance_analysis_workflow2 import run_instance_analysis2


def run_all(input_folder, n_jobs, reorder, gpu_id):
    tpa1 = run_pixel_analysis(input_folder, None, n_jobs, reorder)
    tia1 = run_instance_analysis1(input_folder, None, n_jobs, reorder, gpu_id)
    tia2 = run_instance_analysis2(input_folder, None, n_jobs, reorder, gpu_id, False)

    print("Run pixel analysis in", tpa1)
    print("Run instance analysis 1 in", tia1)
    print("Run instance analysis 2 in", tia2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pixel analysis workflow')
    parser.add_argument('input_folder', type=str, help='')
    parser.add_argument('--n_jobs', type=int, help='', default=1)
    parser.add_argument('--reorder', type=int, default=1, help='')

    args = parser.parse_args()
    run_pixel_analysis(args.input_folder, args.n_jobs, bool(args.reorder))
