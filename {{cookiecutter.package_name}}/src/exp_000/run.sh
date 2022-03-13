echo "START PREPROCESS --->"
python run_preprocess.py --config_name config.yaml
echo "<--- END PREPROCESS"

echo "START TRAIN --->"
for i in `seq 0 4`
do
    echo "START - FOLD: $i"
    python run_train.py --config_name config.yaml --fold $i

    ret=$?
    if [ $ret -ne 0 ]; then
        echo "RAISED EXCEPTION"
        exit 1
    fi
    echo "END - FOLD: $i"
done
echo "<-- END TRAIN"

echo "START VALID --->"
python run_valid.py --ckpt loss --cam
python run_valid.py --ckpt score --cam
echo "<--- END VALID"


echo "START TEST"
python run_test.py --ckpt loss
python run_test.py --ckpt score
echo "<--- END TEST"

exp_dir=`cat config.yaml | grep -E -o "exp_[0-9]+"`
git add -A .
git commit -m "feat: $exp_dir"
git push
