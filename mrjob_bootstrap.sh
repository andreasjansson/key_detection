#!/bin/bash

sudo easy_install pip
sudo pip install numpy scipy argparse
sudo apt-get install mpg123

yes | sudo apt-get install timidity
sudo sh -c "cat >> /etc/apt/sources.list << EOF
deb http://www.debian-multimedia.org squeeze main non-free
deb http://www.debian-multimedia.org testing main non-free
EOF"
gpg --keyserver hkp://pgpkeys.mit.edu --recv-keys 07DC563D1F41B907
gpg --armor --export 07DC563D1F41B907 | sudo apt-key add -
sudo apt-get update
sudo apt-get -y --force-yes install lame libmp3lame-dev faad

sudo pip install boto
sudo sh -c "cat > /etc/boto.cfg <<EOF
[Credentials]
aws_access_key_id = XXXXX
aws_secret_access_key = XXXXX
EOF"
