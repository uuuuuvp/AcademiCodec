# Codec: wish we could be better

## On going
This project is on going. You can find the paper on https://arxiv.org/pdf/2305.02765.pdf <br/>
Furthermore, this project is lanched from University, we expect more researchers to be the contributor. <br/>


## News
### Codec
- 2023.6.2: Add `HiFi-Codec-24k-320d/infer.ipynb`, which can be used to infer acoustic tokens to use for later training of VALL-E, SoundStorm and etc.
- 2023.06.13: Refactor the code structure.
### Dependencies
* [PyTorch](http://pytorch.org/) version >= 1.13.0
* Python version >= 3.8

## Training or Inferce
Refer to the specical folders, e.g. Encodec_24k_240d represent, the Encodec model, sample rate is 24khz, downsample rate is 240. If you want to use our pre-trained models, please refer to https://huggingface.co/Dongchao/AcademiCodec/tree/main