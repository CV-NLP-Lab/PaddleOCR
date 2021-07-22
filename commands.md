python3 -m paddle.distributed.launch --gpus '0'  tools/train.py -c configs/rec/srn_ic15.yml
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/srn_ic15.yml -o Global.checkpoints=./output/rec/srn_new/best_accuracy
python3 tools/infer_rec.py -c configs/rec/srn_ic15.yml -o Global.pretrained_model=./pretrain_models/rec_r50_vd_srn_train/best_accuracy Global.load_static_weights=false Global.infer_img=doc/imgs_words/en/word_1.png
python3 tools/infer_rec.py -c configs/rec/srn_lmdb.yml -o Global.pretrained_model=./output/rec/srn_lmdb/iter_4000 Global.load_static_weights=false Global.infer_img=./test_data/rec/ic15_data/test/word_264.png

