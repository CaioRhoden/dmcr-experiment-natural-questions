conda activate nq

python create_indeces.py --indice_name bm25
python create_indeces.py --indice_name bge
python create_indeces.py --indice_name nv2
python create_indices.py --indice_name qwen

python run_rag.py --retriever bge --vector_db_path data/indices/bge_index.faiss --embedder_path models/bge-base-en-v1.5
python run_rag.py --retriever bge --vector_db_path data/indices/bge_index.faiss --embedder_path models/bge-base-en-v1.5 --nprobe 64
python run_rag.py --retriever bge --vector_db_path data/indices/bge_index.faiss --embedder_path models/bge-base-en-v1.5 --nprobe 1

