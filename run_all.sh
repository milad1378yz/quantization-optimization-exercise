rm -r results/*

python src/main.py --vector_size large --use_multiprocessing
python src/main.py --vector_size small --use_multiprocessing
python src/main.py --vector_size large 
python src/main.py --vector_size small 

# check if GPU is available
python src/main.py --vector_size large --device cuda
python src/main.py --vector_size small --device cuda
python src/main.py --vector_size large --device cuda --use_multiprocessing
python src/main.py --vector_size small --device cuda --use_multiprocessing
