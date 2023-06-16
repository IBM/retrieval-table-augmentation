#!/usr/bin/env bash

#stop at first error, unset variables are errors
set -o nounset
set -o errexit

for NUM in {0..9}
do
  echo "On part ${NUM}"
  wget http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/0${NUM}.tar.gz
  tar -xf 0${NUM}.tar.gz
  lbzip2 0${NUM}.tar
  rm 0${NUM}.tar.gz
done

for NUM in {10..50}
do
  echo "On part ${NUM}"
  wget http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/${NUM}.tar.gz
  tar -xf ${NUM}.tar.gz
  lbzip2 ${NUM}.tar
  rm ${NUM}.tar.gz
done
