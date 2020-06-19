import json

PLATE_NAMES = [
    "20200405_test_images",
    "20200406_164555_328",
    "20200406_210102_953",
    "20200406_222205_911",
    "20200410_145132_254",
    "20200410_172801_461",
    "20200415_150710_683",
    "20200417_132123_311",
    "20200417_152052_943",
    "20200417_172611_193",
    "20200417_185943_790",
    "20200417_203228_156",
    "20200420_152417_316",
    "20200420_164920_764",
    "plate1rep3_20200505_100837_821",
    "plate2rep3_20200507_094942_519",
    "plate5rep3_20200507_113530_429",
    "plate6rep2_wp_20200507_131032_010",
    "plate7rep1_20200426_103425_693",
    "plate8rep1_20200425_162127_242",
    "plate8rep2_20200502_182438_996",
    "plate9_2rep1_20200506_163349_413",
    "plate9_2rep2_20200515_230124_149",
    "plate9_3rep1_20200513_160853_327",
    "plate9rep1_20200430_144438_974",
    "plateK10rep1_20200429_122048_065",
    "plateK11rep1_20200429_140316_208",
    "plateK12rep1_20200430_155932_313",
    "plateK13rep1_20200430_175056_461",
    "plateK14rep1_20200430_194338_941",
    "plateK15rep1_20200502_134503_016",
    "plateK16rep1_20200502_154211_240",
    "plateK17rep1_20200505_115526_825",
    "plateK18rep1_20200505_134507_072",
    "PlateK19rep1_20200506_095722_264",
    "PlateK20rep1_20200506_114059_490",
    "PlateK21rep1_20200506_132517_049",
    "plateK22rep1_20200509_094301_366",
    "plateK23rep1_20200512_103139_970",
    "plateK25rep1_20200512_123527_554",
    "plateK26rep1_20200515_221809_658",
    "plateT1rep1_20200509_114423_754",
    "plateT2rep1_20200509_190719_179",
    "plateT3rep1_20200509_152617_891",
    "plateT4rep1_20200509_171215_610",
    "plateT5rep1_20200512_143609_835",
    "plateT6_20200513_105342_945",
    "plateT7_20200513_131739_093",
    "plateT8rep1_20200516_091304_432",
    "plateU13_T9rep1_20200516_105403_122",
    "plateU1rep1_20200519_102648_382",
    "plateU2rep1_20200519_120202_377",
    "plateU3rep1_20200519_134143_061",
    "plateU4rep1_20200519_155958_647",
    "plateU5rep1_20200519_173432_260",
    "plateU6rep1_20200519_192153_222",
    "plateU7rep1_20200519_210009_665",
    "plateU8rep1_20200519_223701_819",
    "titration_plate_20200403_154849",
    "test", "scheme1", "scheme2", "scheme3", "scheme4"
]


def has_two_bg_wells(name):
    isT = 'plateT' in name
    isU = 'plateU' in name

    is_recentK = False
    if 'plateK' in name or 'PlateK' in name:
        prelen = len('plateK')
        kid = int(name[prelen:prelen+2])
        if kid >= 22:
            is_recentK = True

    return isT or is_recentK or isU


def get_bg_well(name):
    if has_two_bg_wells(name):
        return ['H01', 'G01']
    else:
        return ['H01']


def update_bg_dict():

    with open('./plates_to_background_well.json') as f:
        old_bg_dict = json.load(f)

    default_fixed_bg = {'serum_IgA': 1800., 'serum_IgG': 1300.}
    special_bgs = {
        "20200417_132123_311": {"serum_IgG": 1300, "serum_IgA": 1700},
        "20200417_152052_943": {"serum_IgG": 1500, "serum_IgA": 1650},
        "20200420_164920_764": {"serum_IgG": 1300, "serum_IgA": 1700},
        "20200420_152417_316": {"serum_IgG": 1450, "serum_IgA": 1700},
        "plate7rep1_20200426_103425_693": {"serum_IgG": 1600, "serum_IgA": 2300},
        "plate8rep1_20200425_162127_242": {"serum_IgG": 1500, "serum_IgA": 2300},
        "plate8rep2_20200502_182438_996": {"serum_IgG": 1500, "serum_IgM": "plates/background"},
        "plate9rep1_20200430_144438_974": {"serum_IgG": 1500, "serum_IgA": 2000},
        "plateK10rep1_20200429_122048_065": {"serum_IgG": 1300, "serum_IgA": 2000},
        "plateK11rep1_20200429_140316_208": {"serum_IgG": 1300, "serum_IgA": 2000},
        "plateK15rep1_20200502_134503_016": {"serum_IgG": 1500, "serum_IgA": 2000},
        "plateK16rep1_20200502_154211_240": {"serum_IgG": 1500, "serum_IgA": 2000},
        "plate6rep2_wp_20200507_131032_010": {'serum_IgA': 7000, 'serum_IgG': 5000}
    }

    bg_dict = {}
    for plate_name in PLATE_NAMES:
        if plate_name in special_bgs:
            bg_dict[plate_name] = special_bgs[plate_name]
        elif plate_name in old_bg_dict:
            bg_dict[plate_name] = default_fixed_bg
        else:
            bg_dict[plate_name] = get_bg_well(plate_name)

    with open('./plates_to_background_well_new.json', 'w') as f:
        json.dump(bg_dict, f, sort_keys=True, indent=2)


update_bg_dict()
