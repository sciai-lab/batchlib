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

## Replication
TBD

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
  "result_tables": [
    {
      "analysis_name": "analysis1_table",
      "tables": [
        {
          "table_name": "wells/default",
          "results": [
          
          ]
        }, 
        {
          "table_name": "images/default",
          "results": [
          
          ]
        },
        {
          "table_name": "cells/default",
          "results": [
          
          ]
        }
      ]
    }
  ]
}
```

