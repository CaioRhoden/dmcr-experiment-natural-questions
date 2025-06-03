# context_window_aalysis

**Objective**: This has the motivation to watch the behavior of the Datamodels vs RAG generations when applying it to different sizes of context windows. Currently we're not sure how the context size affects the recall of answers and how the selection through datamodels impact in the downstream pipeline task

**Setup**: The setup of test wil be using four different windows of context: [4,8,16,32] in two different language models (Lllama-3.2-8B-Instruct and Qwen3-8B). For this test we will a subset of 50 samples of the test dataset (same as the *subpartition_rag_datamodels*). The similarity score will be Inner Product using the bte-1.5-base-en encodder with FAISS Index (highest performance in preivous experiments). The pipeline parameters are:

TODO: Fill parameters

**Folder organization**: We will have two main folders, *llama* and *qwen*, inside of both will be present four folders: *k_4*, *k_8*, *k_16* and *k_32*. For each of this subfolders will be all the configs and pipeline results.

**Results**: TODO