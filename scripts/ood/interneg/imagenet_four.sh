tau=1.0
prompt=nice
random_permute=True
in_score=sum



for seed in 0
do
    for backbone in ViT-B/16 # RN50 RN101 ViT-B/32 ViT-L/14 ViT-B/16
    do
        for group_num in 5 #1 5 10 50 100 200 300 400 500 1000
        do
            for thres in 0.35 # 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.8 1.0
            do
                for extra_text_length in 2000 # 100 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 50000
                do
                    for text_length in 2000 # 100 500	1000 2000 5000 10000 20000	50000
                    do
                        new_length=$((text_length + 1000))
                        python main.py \
                        --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
                        configs/networks/fixed_clip.yml \
                        configs/pipelines/test/test_fsood.yml \
                        configs/preprocessors/base_preprocessor.yml \
                        configs/postprocessors/mcm.yml \
                        --dataset.test.batch_size 256 \
                        --ood_dataset.batch_size 256 \
                        --dataset.num_classes ${new_length} \
                        --ood_dataset.num_classes ${new_length}  \
                        --evaluator.name ood_clip_tta \
                        --network.name fixedclip_inter_negoodprompt \
                        --network.backbone.ood_number ${text_length} \
                        --network.backbone.name  ${backbone} \
                        --network.backbone.text_prompt ${prompt} \
                        --network.backbone.text_center True \
                        --network.pretrained False \
                        --postprocessor.APS_mode False \
                        --postprocessor.name oneoodpromptdevelop \
                        --postprocessor.postprocessor_args.group_num ${group_num}  \
                        --postprocessor.postprocessor_args.random_permute ${random_permute}  \
                        --postprocessor.postprocessor_args.extra_text_length ${extra_text_length} \
                        --postprocessor.postprocessor_args.thres ${thres}  \
                        --postprocessor.postprocessor_args.tau ${tau}  \
                        --postprocessor.postprocessor_args.in_score ${in_score}  \
                        --num_gpus 1 --num_workers 6 \
                        --seed ${seed} \
                        --merge_option merge \
                        --output_dir ./Ours_result/ \
                        --mark FourOOD_backbone_${backbone}_text_length${text_length}_seed${seed}_thres${thres}_extra_text_length${extra_text_length}_group_num_${group_num}_random_${random_permute}_official
                    done
                done
            done
        done
    done
done

