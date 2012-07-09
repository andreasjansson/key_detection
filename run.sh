#!/bin/bash

if [[ $1 == 'dev' ]]
then

    python mr_train.py filenames.data > output_dev

    exit 0
fi

#fi619425

tar czvf util.tar.gz util.py 
#python mr_train.py --num-ec2-instances 5 --python-archive util.tar.gz -r emr 's3://andreasjansson/beatles/filenames.data' > output

python mr_train.py --num-ec2-instances 1 --file filenames.data --python-archive util.tar.gz -r emr filenames.data > output
