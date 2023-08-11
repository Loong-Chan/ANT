# Photo
python main.py --dataset photo --backbone GCN --n_cat 1 --hidden 128 --consis False --n_aug 4 --lr 0.0005 --gnn_epoch 500 --weight_decay 0.0005
python main.py --dataset photo --backbone GAT --n_cat 2 --hidden 64 --consis False --n_aug 4 --lr 0.00003 --gnn_epoch 500 --weight_decay 0.001
python main.py --dataset photo --backbone GATv2 --n_cat 2 --hidden 64 --consis False --n_aug 4 --lr 0.00005 --gnn_epoch 400 --weight_decay 0.001
python main.py --dataset photo --backbone MLP --n_cat 1 --hidden 1024 --consis False --n_aug 4 --lr 0.0001 --gnn_epoch 200 --weight_decay 0.000001
python main.py --dataset photo --backbone S2GC --n_cat 1 --hidden 64 --consis False --n_aug 4 --lr 0.001 --gnn_epoch 500 --weight_decay 0.05
python main.py --dataset photo --backbone GCNII --n_cat 1 --hidden 16 --consis False --n_aug 4 --lr 0.005 --gnn_epoch 1000 --weight_decay 0.00001

# Cora
python main.py --dataset cora --backbone MLP --n_cat 1 --hidden 1024 --consis True --n_aug 4 --lr 0.001 --gnn_epoch 200 --weight_decay 0.002
python main.py --dataset cora --backbone GCN --n_cat 1 --hidden 64 --consis True --n_aug 4 --lr 0.01 --gnn_epoch 500 --weight_decay 0.00005
python main.py --dataset cora --backbone GATv2 --n_cat 1 --hidden 16 --consis True --n_aug 4 --lr 0.0005 --gnn_epoch 500 --weight_decay 0.0002
python main.py --dataset cora --backbone GAT --n_cat 1 --hidden 64 --consis True --n_aug 4 --lr 0.001 --gnn_epoch 200 --weight_decay 0.001
python main.py --dataset cora --backbone S2GC --n_cat 1 --hidden 64 --consis True --n_aug 4 --lr 0.00005 --gnn_epoch 500 --weight_decay 0.0001
python main.py --dataset cora --backbone GCNII --n_cat 1 --hidden 32 --consis True --n_aug 4 --lr 0.005 --gnn_epoch 500 --weight_decay 0.0002
