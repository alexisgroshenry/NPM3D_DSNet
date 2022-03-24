ngpu=1
tag=test_dsnet_pytorch_dist

python -m torch.distributed.launch --nproc_per_node=${ngpu} cfg_train.py \
    --tcp_port 12345 \
    --batch_size ${ngpu} \
    --config cfgs/release/dsnet.yaml \
    --pretrained_ckpt pretrained_weight/dsnet_pretrain_pq_0.577.pth \
    --tag ${tag} \
    --launcher pytorch \
    --fix_semantic_instance \
    --onlytest