<div align="center">
    <h2>torchrecurrent</h2>
</div>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/torchrecurrent.svg)](https://pypi.org/project/torchrecurrent/)
[![codecov](https://codecov.io/gh/MartinuzziFrancesco/torchrecurrent/graph/badge.svg?token=AW36UWD1OM)](https://codecov.io/gh/MartinuzziFrancesco/torchrecurrent)
[![Build](https://github.com/MartinuzziFrancesco/torchrecurrent/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinuzziFrancesco/torchrecurrent/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-gh--pages-blue?logo=github)](https://MartinuzziFrancesco.github.io/torchrecurrent/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

Pytorch compatible implementation of various recurrent layers
found in the literature.
**Disclaimer**: `torchrecurrent` is an independent project
and is not affiliated with the PyTorch project or Meta AI.
The name reflects compatibility with PyTorch, not any official endorsement.

## Installation

```shell
pip install torchrecurrent
```

## Features

| Short name | Publication venue | Official implementation |
|------------|-------------------|-----------------------------|
| [**AntisymmetricRNN/GatedAntisymmetricRNN**](https://arxiv.org/abs/1902.09689) | ICLR 2019 | – |
| [**ATR**](https://arxiv.org/abs/1810.12546) | EMNLP 2018 | [bzhangGo/ATR](https://github.com/bzhangGo/ATR) |
| [**BR/BRC**](https://doi.org/10.1371/journal.pone.0252676) | PLOS ONE 2021 | [nvecoven/BRC](https://github.com/nvecoven/BRC) |
| [**CFN**](https://arxiv.org/abs/1612.06212) | ICLR 2017 | – |
| [**coRNN**](https://arxiv.org/abs/2010.00951) | ICLR 2021 | [tk-rusch/coRNN](https://github.com/tk-rusch/coRNN) |
| [**FastRNN/FastGRNN**](https://arxiv.org/abs/1901.02358) | NeurIPS 2018 | [Microsoft/EdgeML](https://github.com/Microsoft/EdgeML) |
| [**FSRNN**](https://arxiv.org/abs/1705.08639) | NeurIPS 2017 | [amujika/Fast-Slow-LSTM](https://github.com/amujika/Fast-Slow-LSTM) |
| [**IndRNN**](https://arxiv.org/abs/1803.04831) | CVPR 2018 | [Sunnydreamrain/IndRNN_Theano_Lasagne](https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne) |
| [**JANET**](https://arxiv.org/abs/1804.04849) | arXiv 2018 | [JosvanderWesthuizen/janet](https://github.com/JosvanderWesthuizen/janet) |
| [**LEM**](https://arxiv.org/pdf/2110.04744) | ICLR 2022 | [tk-rusch/LEM](https://github.com/tk-rusch/LEM) |
| [**LiGRU**](https://arxiv.org/abs/1803.10225) | IEEE Transactions on Emerging Topics in Computing 2018 | [mravanelli/theano-kaldi-rnn](https://github.com/mravanelli/theano-kaldi-rnn/) |
| [**LightRU**](https://www.mdpi.com/2079-9292/13/16/3204) | MDPI Electronics 2023 | – |
| [**MinimalRNN**](https://arxiv.org/abs/1711.06788) | NeurIPS 2017 | – |
| [**MultiplicativeLSTM**](https://arxiv.org/abs/1609.07959) | Workshop ICLR 2017 | [benkrause/mLSTM](https://github.com/benkrause/mLSTM) |
| [**MGU**](https://arxiv.org/abs/1603.09420) | International Journal of Automation and Computing 2016 | – |
| [**MUT1/MUT2/MUT3**](https://proceedings.mlr.press/v37/jozefowicz15.pdf) | ICML 2015 | – |
| [**NAS**](https://arxiv.org/abs/1611.01578) | arXiv 2016 | [tensorflow_addons/rnn](https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/rnn/nas_cell.py#L29-L236) |
| [**OriginalLSTM**](https://ieeexplore.ieee.org/abstract/document/6795963) | Neural Computation 1997 | - |
| [**PeepholeLSTM**](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf) | JMLR 2002 | – |
| [**RAN**](https://arxiv.org/abs/1705.07393) | arXiv 2017 | [kentonl/ran](https://github.com/kentonl/ran) |
| [**RHN**](https://arxiv.org/abs/1607.03474) | ICML 2017 | [jzilly/RecurrentHighwayNetworks](https://github.com/jzilly/RecurrentHighwayNetworks) |
| [**SCRN**](https://arxiv.org/abs/1412.7753) | ICLR 2015 | [facebookarchive/SCRNNs](https://github.com/facebookarchive/SCRNNs) |
| [**SGRN**](https://doi.org/10.1049/gtd2.12056) | IET 2018 | – |
| [**STAR**](https://arxiv.org/abs/1911.11033) | IEEE Transactions on Pattern Analysis and Machine Intelligence 2022 | [0zgur0/STAckable-Recurrent-network](https://github.com/0zgur0/STAckable-Recurrent-network) |
| [**Typed RNN / GRU / LSTM**](https://arxiv.org/abs/1602.02218) | ICML 2016 | – |
| [**UGRNN**](https://arxiv.org/abs/1611.09913) | ICLR 2017 | - |
| [**UnICORNN**](https://arxiv.org/abs/2103.05487) | ICML 2021 | [tk-rusch/unicornn](https://github.com/tk-rusch/unicornn) |
| [**WMCLSTM**](https://arxiv.org/abs/2109.00020) | Neural Networks 2021 | – |

## See also

[LuxRecurrentLayers.jl](https://github.com/MartinuzziFrancesco/LuxRecurrentLayers.jl):
Provides recurrent layers for Lux.jl in Julia.

[RecurrentLayers.jl](https://github.com/MartinuzziFrancesco/RecurrentLayers.jl):
Provides recurrent layers for Flux.jl in Julia.


[ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl):
Reservoir computing utilities for scientific machine learning.
Essentially gradient free trained recurrent neural networks.

## License

This project’s own code is distributed under the MIT License (see [LICENSE](LICENSE)). The primary intent of this software is academic research.

### Third-party Attributions

Some cells are re-implementations of published methods that carry their own licenses:
- **NASCell**: originally available under Apache 2.0 — see [LICENSE-Apache2.0.txt](licenses/Apache2.0.txt).

Please consult each of those licenses for your obligations when using this code in commercial or closed-source settings.
