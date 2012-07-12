#!/bin/bash

if [[ $1 == 'dev' ]]
then

    python mr_train.py filenames.data > output_dev

    exit 0
fi

data=filenames.lily50.data

tar czvf util.tar.gz util.py 
python mr_train.py --num-ec2-instances 5 --file $data --python-archive util.tar.gz -r emr $data > output
