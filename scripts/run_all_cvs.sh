python run_models.py --mode cv --save True --feature main --device cuda:0 --n 30
python run_models.py --mode cv --save True --feature no_labs --device cuda:0 --n 30
python run_models.py --mode cv --save True --feature simple --device cuda:0 --n 30
python run_models.py --mode cv --save True --feature genes_only --device cuda:0 --n 30
python run_models.py --mode cv --save True --feature genes_cancer_type --device cuda:0 --n 30
python run_models.py --mode cv --save True --feature cancer_type_only --device cuda:0 --n 30
python run_models.py --mode cv --save True --feature basic_chemo --device cuda:0 --n 30
python run_models.py --mode cv --save True --feature basic_genetic --device cuda:0 --n 30
python run_models.py --mode cv --save True --feature basic_binarized --device cuda:0 --n 30
python run_models.py --mode cv --save True --feature ext --device cuda:0 --n 30
