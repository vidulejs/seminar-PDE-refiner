python scripts/train.py -c ./configs/kuramotosivashinsky1d_UNET.yaml --data.data_dir /media/dan/DATA1/Kuramoto-Sivashinsky-Fixed
python scripts/train.py -c ./configs/kuramotosivashinsky1d_FNO.yaml --data.data_dir /media/dan/DATA1/Kuramoto-Sivashinsky-Fixed
python scripts/pderefiner_train.py -c ./configs/kuramotosivashinsky1d.yaml --data.data_dir /media/dan/DATA1/Kuramoto-Sivashinsky-Fixed