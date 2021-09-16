# Code Snippets

## MacOS Terminal CLI

### Remove directory with files

```bash
rm -R <directory_name>
```

### Copy Folder to another directory

```bash
cp -R <source directory path> <target directory path> 
```

Example

`cp ./Riskscore ./SageMaker/Riskscore`

Note, no slashes after the directory name.  And you need to assign the directory name to the target destination.

## AWS cli

### Updating aws cli

`pip install awscli --upgrade --user`

### Recursively delete items in a s3 folder

`aws s3 rm --recursive s3://your_bucket_name/foo/`

### Add regular expressions

`aws s3 rm <your_bucket_name> --exclude "*" --include "*137ff24f-02c9-4656-9d77-5e761d76a273*"`

### List Items in a S3 bucket

`aws s3 ls <s3_uri>` Don't include an s3_uri if you want aws to list all files in the account

### Put Objects in S3 bucket

`aws s3 put-object --bucket <bucket_name> --key <folder/>`
`aws s3 put-object --bucket <bucket_name> --key <file>`

## SageMaker SDK

### Displaying list of files in a Bucket/folder

```python
#Purpose: Displaying list of files in a Bucket/folder
import os
import boto3
import pandas as pd

data_bucket = 'prod-ba-research-datalake'
prefix = 'gyroscope/prod/client/'
client = 'txdallaspd'
extract_type = 'ml' # Or table
extraction_date = '2021-07-28' 
s3_client = boto3.client('s3')

data_directory = os.path.join(prefix, client, extract_type)

response = s3_client.list_objects(
            Bucket=data_bucket,
            Prefix = data_directory,
            MaxKeys = 12)
pd.DataFrame.from_dict(response['Contents'])
```

Prints a dictionary of the results at the end.

### Create folders inside of a bucket

```python
# Create folders inside of a bucket
import boto3

s3_client = boto3.client('s3')
folder_names = ['Bob.Bell','Marcus.Hilliard']
for folder_name in folder_names:
    s3_client.put_object(Bucket=bucket, Key=(folder_name+'/'))
```

### Copy Files Between Buckets

```python
# Copy Files between buckets
import boto3

s3 = boto3.resource('s3')
copy_source = {
    "Bucket" : "ba-research-dev",
    "Key": "zfreundt/nashville/"
}
s3.meta.client.copy(copy_source, 'ba-research-dev', 'mhilliard/nashville/')
```

## Python

### Pandas

#### Pandas Display Settings

```python
# Pandas Settings
# Purpose: Setting dataframe display setting.
pd.options.display.max_colwidth = 100
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

#### Dictionary to DataFrame

```python
# Purpose: Convert dictionary to a pandas DataFrame
pd.Dataframe.from_dict(<dictionary>)
```

#### Creating empty dataframe with column names set

```python
# Purpose: Create an empty dataframe, but name columns
row_counts = pd.DataFrame(columns = ['Filename','Row Counts'])
```

#### Extrating a file name from a path

```python
# To extract a file name from a path.  It returns the file name as a string
filename = os.path.basename("path/to/file/sample.txt")
print(filename)
```
