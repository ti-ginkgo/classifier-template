#!/bin/bash

echo "START PREPROCESS --->"
python run_preprocess.py --config_name config.yml
echo "<--- END PREPROCESS"

echo "START TRAIN --->"
for i in `seq 0 4`
do
    echo "START - FOLD: $i"
    python run_train.py --config_name config.yml --fold $i

    ret=$?
    if [ $ret -ne 0 ]; then
        echo "RAISED EXCEPTION"
        exit 1
    fi
    echo "END - FOLD: $i"
done
echo "<-- END TRAIN"

echo "START OOF --->"
python run_oof.py --ckpt loss --cam
python run_oof.py --ckpt score --cam
echo "<--- END OOF"

echo "START INFERENCE PREPROCESS --->"
python run_inference_preprocess.py --config_name config.yml
echo "<--- END INFERENCE PREPROCESS"

echo "START INFERENCE"
python run_inference.py --ckpt loss
python run_inference.py --ckpt score
echo "<--- END INFERENCE"

exp_dir=`cat configs/config.yml | grep -E -o "exp_[0-9]+"`
git add -A .
git commit -m "feat: $exp_dir"
git push
