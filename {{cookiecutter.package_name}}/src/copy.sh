src=$1
target=$2

mkdir $target
cp $src/*.py $target
cp $src/*.sh $target
cp -r $src/configs $target
