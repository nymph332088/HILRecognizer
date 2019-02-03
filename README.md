# HILRecognizer
Code to reproduce results of KDD 2019


# Get Started

HILRecognizer contains code and datasets for KDD 2019 paper:

* **Zhang, S., He, L., Vucetic, S., Dragut, E., Human-in-the-loop ML Systems for Entity Extraction, KDD, 2019**

To run the code, the following environment is required:
* python==2.7.6
* torch==0.3.1

# Run 5 fold cross validation. 
The 5-fold cross validation is used to select the best hyperparameters based on the weaklly labelled data, via ``random search`` technique. 
After the 5 fold cross validation, the best hyperparameter ``XX.pkl`` is output to the ``experimentas/kaggle`` folder.

``
CUDA_VISIBLE_DEVICES="$dev" python s_train_bilstm_tagger.py --data data/position/testKaggleAll.csv \
--save experiments/position/kaggle_bound/pretrain \
--params experiments/position/kaggle_bound/loo_R.E.tag_best_args.pkl \
--epochs 5 --cuda --batch-size 512 --tags o y --max_len 104 --label R.E.tag --run pretrain``

# Pretrain on weakly labelled data

``CUDA_VISIBLE_DEVICES="$dev" python s_train_bilstm_tagger.py --data data/position/testKaggleAll.csv \
--save experiments/position/kaggle_bound/pretrain \
--params experiments/position/kaggle_bound/loo_R.E.tag_best_args.pkl \
--epochs 5 --cuda --batch-size 512 --tags o y --max_len 104 --label R.E.tag --run pretrain``

# Fine-tuninig pre-trained model with active learning
``
aliter=50 # active learning iterations
albs=20   # active learning batch size
epoch=10  # active learning epochs
best_args="loo_R.E.tag_best_args.pkl"   # best args of pretrained model
pretrain= "R.E.tag_testKaggleAll_pretrain_5.pt"   # pretrained model
outfolder="active_learning_cv_by_outlet_retag_pt5" # output folder
CUDA_VISIBLE_DEVICES="$dev" python s_train_bilstm_tagger.py --data data/position/testKaggle2.csv \
--save experiments/position/kaggle_bound/"$outfolder"/ \
--params experiments/position/kaggle_bound/"$best_args" \
--pretrain experiments/position/kaggle_bound/pretrain/"$pretrain" \
--epochs 10 --cuda --partition outlet --batch-size 300 --tags o y --max_len 104 --label TagLabel \
--fold "$fold" --run al --al_bs "$albs" --al_iter "$aliter" ``