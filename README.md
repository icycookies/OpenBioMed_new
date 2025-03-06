### Installation
```
conda create -n OpenBioMed python=3.9
conda activate OpenBioMed
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117  
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install pytorch_lightning==2.0.8 peft==0.9.0 accelerate==1.3.0 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```



Optional:
```
# For visualization
conda install -c conda-forge pymol-open-source
pip install imageio

# For AutoDockVina
pip install meeko==0.1.dev3 pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# For overlap-based evaluation
pip install spacy rouge_score nltk
python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
```