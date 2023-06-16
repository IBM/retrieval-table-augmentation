Download the 2015 English relational tables:
```bash
wget http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/${NUM}.tar.gz
tar -xf ${NUM}.tar.gz
lbzip2 ${NUM}.tar
rm ${NUM}.tar.gz
```
NUM from 00 to 50

```bash
bash ${PYTHONPATH}/table_augmentation/webtables/webtable_download.sh
```

