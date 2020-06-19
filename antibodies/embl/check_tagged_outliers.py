import os
from glob import glob
from process_for_manuscript import all_kinder_plates, all_manuscript_plates


def check_missing_outliers():
    all_plates = all_kinder_plates() + all_manuscript_plates()
    outlier_pattern = '../../misc/tagged_outliers/*.csv'

    outlier_plates = glob(outlier_pattern)
    outlier_plates = [os.path.split(plate)[1] for plate in outlier_plates]
    outlier_plates = [plate[:plate.find('_tagger')] for plate in outlier_plates]

    missing_plates = list(set(all_plates) - set(outlier_plates))
    missing_plates.sort()
    print("Missing tagged outliers")
    print("\n".join(missing_plates))


if __name__ == '__main__':
    check_missing_outliers()
