# How to run the production pipeline at EMBL

1. You need a `batchlib` clone on current master. You can either use the version at `/g/kreshuk/software/batchlib` or use your own.
2. Log in to `login.cluster.embl.de` via `ssh USER@login.cluster.embl.de` (only accessible from the EMBL VPN).
2. Go to `batchlib/antibodies/embl`
3. Make a json file containing a list with the plates that should be run, e.g. `["/g/kreshuk/data/covid/covid-data-vibor/tiny-test"]`
4. Submit all the plates for processing via `./submit_folders.py plates.json --config_file configs/cell_analysis_db.conf`
5. Once all computations have finished, you can upload the results via `./summarize_results.py --inputs new.json --host vm-kreshuk-11.embl.de --password <DB_PASSWORD> --token <SLACK_TOKEN>`
