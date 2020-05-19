# MongoDB setup

## Installation

Install MongoDB following the [instructions](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu)

If MongoDB is installed with the package manager the default data directory will be `/var/lib/mongodb`, logs
 will be stored in `/var/log/mongodb`

The location of the default config file: `/etc/mongod.conf`

### Start MongoDB

Start MongoDB primary instance

```bash
sudo systemctl start mongod
```

and verify that it started successfully

```bash
sudo systemctl status mongod
```
    

### Setup access control and covid DB

- create the user administrator

In the mongo shell execute:
```
use admin
db.createUser(
  {
    user: "admin",
    pwd: passwordPrompt(),
    roles: [ { role: "userAdminAnyDatabase", db: "admin" }, "readWriteAnyDatabase" ]
  }
)
```

- enable authentication in the config

Add the follwing to `/etc/mongod.conf`
```
security:
    authorization: enabled
```

and restart mongod service

- create `covid` DB and `covid19` user 

Start mongo shell and authenticate as admin

```
use admin // switch to admin db
db.auth("admin", passwordPrompt()) // or cleartext password
```

Create create `covid` db and `covid19` user

```
use covid
db.createUser(
  {
    user: "covid19",
    pwd:  passwordPrompt(), // or cleartext password
    roles: [ { role: "readWrite", db: "covid" }]
  }
)
```

Connect as `covid19` user and create a sample document to test the setup

```bash
mongo --port 27017 -u "covid19" --authenticationDatabase "covid" -p
```

```
use covid
db.foo.insert( { x: 1, y: 1 } )
```

## Schema description
The following collections are present in the `covid` database:
- `immuno-assay-metadata` -  a collection to store all metadata associated with the plates

Schema of a single document:
```json
{
  "created_at": "TIMESTAMP",
  "name": "PLATE NAME; UNIQUE ID", 
  "outlier":  "QUALITY CONTROL FLAG",
  "outlier_type": "QUALITY CONTROL TYPE",
  "channel_mapping": "sub-document containing channel mapping info",
  "wells": [
    {
      "name": "WELL NAME",
      "outlier": "QUALITY CONTROL FOR THE WELL",
      "outlier_type": "QUALITY CONTROL TYPE",
      "manual_assessment": "positive/negative sample assessed manually",
      "images": [
        {
          "name": "IMAGE NAME",
          "well_name": "WELL NAME",
          "outlier": "QUALITY CONTROL: outlier or valid  or none",
          "outlier_type": "additional info about the outlier status",
          "issue_urls (optional)": ["issue-url1", "issue-url2"],
          "inst_seg_gt (optional)": "TRUE/FALSE, whether we have inst seg GT for this image",
          "sem_seg_gt (optional)": "TRUE/FALSE, whether we have sem seg GT for this image"
        },
      ]
    },
  ] 
}
```
Attributes marked as `(optional)` are created on demand and are not required to be present in all documents.

- `immuno-assay-analysis-results` - a collection to store results from our analysis pipeline

Schema of a single document:
```json
{
  "created_at": "TIMESTAMP",
  "workflow_duration": "SECONDS",
  "workflow_name": "name of the workflow",
  "plate_name": "name of the plate",
  "batchlib_version": "version of batchlib the that produced the result",
  "analysis_parameters": "parameters/values the analysis was run with",
  "result_tables": [
    {
      "table_name": "wells/default",
      "results": [
      
      ]
    }, 
    {
      "table_name": "images/default",
      "results": [
      
      ]
    }
  ]
}
```

- `cohort-descriptions` - cohort types together with their short descriptions

Schema of a single document:
```json
{
  "patient_type": "STRING",
  "description": "STRING"
}
```

## Import metadata manually
Attributes such as cohort ids and Elise test results for the wells as well as outlier status for images are provided externally
via excel sheets and CSV files. Those attributes are updated automatically via the [DbResultWriter](result_writer.py) job
which is run at the end of the analysis workflow. Those metadata may change however (e.g. outliers were redone manually,
or new elisa excel sheets were added to the repo). In this case we might want to update the metadata manually, instead
of re-running the analysis workflow on all of the plates.

The following scripts can be run to update the metadata at any time against the dev and production DB:
```bash
python import_outliers.py --host DBHOST --port 27017 --db covid --user covid19 --password PASSWD
python import_cohort_ids.py --host DBHOST --port 27017 --db covid --user covid19 --password PASSWD
python import_elisa_results.py --host DBHOST --port 27017 --db covid --user covid19 --password PASSWD
```

Bear in mind that cohort ids have to be imported before elisa test results, cause the latter rely on the former to be in DB.

## Backup & restore

In order to do periodic backups or backup data from the development database and restore it in the production database 
use `mongodump`/`mongorestore` functionality.

### Dump development database and restore into the production database

Dump development db into the `dump` folder in the current directory:
```bash
mongodump --host=vm-kreshuk08.embl.de --port=27017 --db=covid --username=covid19 --password=PASSWD
```


Restore the dump to the production mongod instance:
```bash
mongorestore --host=vm-kreshuk-11.embl.de --port=27017 --username=covid19  --password=PASSWD --authenticationDatabase=covid dump
```
