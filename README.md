# Dynamic-Struc2vec

Firstly, used requirements.txt to set up venv

src/process.py is used to preprocess the DBLP dataset

preprocessed datas are stored in graph

src/run_all.sh is used to run all static struc2vec pipline

src/run_incremental.sh is used to run all dynamic struc2vec pipline

All embedding results and models are restore in emb

All processed distance graphs are restored in pickles

src/conf/timeline.yml control all parameters in struc2vec algorithm

src/test_all.sh is used to test static struc2vec result

src/test_warm_all.sh is used to test dynamic struc2vec result