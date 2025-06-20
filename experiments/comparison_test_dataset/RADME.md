# comparison_test_dataset

**Objective**: Get the performance of baseline x naive RAG x datamodels in 5 different random seeds

**Setup**: Based on previous results we will truncate some set of parameters. Based on previous results the RAG subset size will be of 100 with the context window of size 16. The main script will create a numpy array with five random ints between 0 and 100 and then save it in the *random_keys.npy*


**Folder organization**: We will have three main folders: *baseline*, *naive_rag* and *datamodels*. Inside each on of them will be different folders containing the generations for each random seed, the folders names will be the previous path plus the index of the random seed beeing used. Example: *naive_rag_2* for the third random seed in the *random_keys.npy* file. Allthe different steps and runs can be executed using the *run_test.py*, all the steps will be chronologically documented in *run.sh*.

**Results**: TODO