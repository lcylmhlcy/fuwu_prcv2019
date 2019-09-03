# fuwu_prcv2019
> 大田作物病害图像识别技术挑战赛（PRCV2019）

## Competition

PRCV2019竞赛通知：http://www.prcv2019.com/竞赛通知.html

大田作物病害图像识别技术挑战赛：http://www.icgroupcas.cn/website_bchtk/fuwu_prcv2019.html

## Dependencies:

+ Ubuntu 16.04
+ cuda 9.0.176 + cudnn 7.5.0
+ Python 3.6
+ PyTorch = 1.1.0

## My Custom Tricks

+ Add Random_PatchShuffle. The idea comes from [JDAI-CV/DCL](https://github.com/JDAI-CV/DCL). 
+ Resize to 200 (width) and 300 (height). Equal scaling can reduce resource consumption without distortion.
+ Add RandomHorizontalFlip and RandomVerticalFlip.
+ Add ColorJitter to make model more robust to light change.
+ Add [Cutout](https://github.com/uoguelph-mlrg/Cutout). 
+ To do: add noise; add Mixup; add [RICAP](https://github.com/4uiiurz1/pytorch-ricap); and so on.

## How to use

### Train
```
python3 main.py --data-root ../data/IDADP --data ImageNet --save ./save1 --arch msdnet --batch-size 32 --epochs 300 --nBlocks 7 --stepmode even --step 4 --base 4 --nChannels 16 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --gpu 0 -j 16
```

### Test
```
python3 predict.py --data-root ../data/IDADP --data ImageNet --save ./save1 --arch msdnet --batch-size 32 --epochs 300 --nBlocks 7 --stepmode even --step 4 --base 4 --nChannels 16 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --evalmode anytime --evaluate-from ./save1/save_models/model_best.pth.tar --gpu 0 -j 16
```

## Acknowledgments

We would like to take immense thanks to [kalviny/MSDNet-PyTorch](https://github.com/kalviny/MSDNet-PyTorch) for providing code, which is a PyTorch implementation of the paper [Multi-Scale Dense Networks for Resource Efficient Image Classification](https://arxiv.org/pdf/1703.09844.pdf)

Citation:

    @inproceedings{huang2018multi,
        title={Multi-scale dense networks for resource efficient image classification},
        author={Huang, Gao and Chen, Danlu and Li, Tianhong and Wu, Felix and van der Maaten, Laurens and Weinberger, Kilian Q},
        journal={ICLR},
        year={2018}
    }