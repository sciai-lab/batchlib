import csv
import glob


def compute_stats(results_directory):
    results = glob.glob(f"{results_directory}/*.csv")
    all_files = []

    for result in results:
        with open(result, 'r') as f:
            csvf = csv.reader(f, delimiter=",")
            for i, line in enumerate(csvf):
                if i != 0:
                    all_files.append(line)

    result = {}

    for evaluation in all_files:
        if evaluation[0] not in result:
            result[evaluation[0]] = [int(evaluation[-1]), 1]
        else:
            if evaluation[-1] != '0':
                result[evaluation[0]][0] += int(evaluation[-1])
                result[evaluation[0]][1] += 1

    sum_ = 0
    print(f'Plate  {40 * " "}Mean Score')
    for key, value in sorted(result.items()):
        print(f'{key}: {(40 - len(key)) * " "} {value[0] / value[1]: .2f}/4  #images {value[1]}')
        sum_ += value[1]
    print(f'# images evaluated: {sum_}')


if __name__ == '__main__':

    compute_stats(results_directory="./results")