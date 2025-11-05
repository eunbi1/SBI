# Environment Setting 
conda create -n sbi python=3.10 -c conda-forge
pip install -r requirements.txt 

# 
cd SBI
conda activate sbi
python main.py kolmogorov --device mps
