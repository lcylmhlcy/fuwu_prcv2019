python3 main.py --data-root ../data/IDADP --data ImageNet --save ./save --arch msdnet --batch-size 32 --epochs 2000 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --gpu 0 -j 16