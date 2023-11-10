# Diversity-Novelty-aware Interactive Recommendation (DNaIR) framework

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/shixiaoyu0216/DNaIR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9.0+cu111-%237732a8)](https://pytorch.org/)

This repository contains the official Pytorch implementation for the paper *Relieving Popularity Bias in Interactive Recommendation: A Diversity-Novelty-Aware Reinforcement Learning Approach*. It also contains the a toy dataset of DNaIR: *Movielens-100K*.

<img src="figs/Framework.png" alt="Framework" style="zoom:100%;" />
<img src="figs/ActionGeneration.png" alt="ActionGeneration" style="zoom:80%;" />

More descriptions are available via the [paper](https://dl.acm.org/doi/10.1145/3618107).

If this work helps you, please kindly cite our papers:

```latex
@article{shi2023relieving,
  title={Relieving Popularity Bias in Interactive Recommendation: A Diversity-Novelty-Aware Reinforcement Learning Approach},
  author={Shi, Xiaoyu and Liu, Quanliang and Xie, Hong and Wu, Di and Peng, Bo and Shang, MingSheng and Lian, Defu},
  journal={ACM Transactions on Information Systems},
  year={2023},
  publisher={ACM New York, NY}
}
```

---
## Installation

1. Clone this git repository and change directory to this repository:

	```bash
	git clone git@github.com:shixiaoyu0216/DNaIR.git
	cd DNaIR
	```


2. A new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) is suggested. 

    ```bash
    conda create --name DNaIR python=3.9 -y
    ```

3. Activate the newly created environment.

    ```bash
    conda activate DNaIR
    ```


4. Install the required 

    ```bash
    pip install -r requirements.txt
    ```


## Download the data

1. Download the compressed dataset

    ```bash 
    wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
	unzip ml-20m.zip -d ml-20m
    ```

 	```bash 
    please go to https://www.kaggle.com/netflix-inc/netflix-prize-data, and
	click 'Download' to get archive.zip. Then place archive.zip in directory
	'conversion_tools/'.
	unzip -o -d ./netflix-data archive.zip
    ```

	```bash 
    wget https://chongming.myds.me:61364/data/KuaiRec.zip --no-check-certificate
 	unzip KuaiRec.zip
    ```

If you use them in your work, please cite: [![LINK](https://img.shields.io/badge/-Paper%20Link-lightgrey)](https://dl.acm.org/doi/abs/10.1145/2827872) [![PDF](https://img.shields.io/badge/-PDF-red)](https://dl.acm.org/doi/abs/10.1145/2827872), [![PDF](https://img.shields.io/badge/-PDF-red)](https://d1wqtxts1xzle7.cloudfront.net/90881302/NetflixPrize-description-libre.pdf?1662854712=&response-content-disposition=inline%3B+filename%3DThe_Netflix_Prize.pdf&Expires=1699603604&Signature=aLzq1fsD73HYHYeZmFOEUOwuaEeR~gWAtj8i7EJMNr0DRXFWckr~ndzyu1zsfWuE4nigx3wAA~WLf-3FqSMk0i9xVVk8T94hcddWs2ILOh4LXsgB8QQa47iJ8Wq1O8Jyecf2gXosxrGXnxACIiBsL7tspTCq4gcKKZudflRp09LuVDGs66rezCHxXRzr~WsQr3siCGY65UKq9sJu~onq0HKA3tROuOJrxWJ~usSGhDw7oSz0QbGlkg5EKtomBIVNGpET0-261YPIy3MpJJQw29sS9FROkbNlA-kKafwbM2dePrd76yr24SePGA6csuHkp6ukYpJ8obAWD4dRLfeTRA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA), [![LINK](https://img.shields.io/badge/-Paper%20Link-lightgrey)](https://arxiv.org/abs/2202.10842) [![PDF](https://img.shields.io/badge/-PDF-red)](https://arxiv.org/pdf/2202.10842.pdf)


```latex
@article{harper2015movielens,
  title={The movielens datasets: History and context},
  author={Harper, F Maxwell and Konstan, Joseph A},
  journal={Acm transactions on interactive intelligent systems (tiis)},
  volume={5},
  number={4},
  pages={1--19},
  year={2015},
  publisher={Acm New York, NY, USA}
}

@inproceedings{bennett2007netflix,
  title={The netflix prize},
  author={Bennett, James and Lanning, Stan and others},
  booktitle={Proceedings of KDD cup and workshop},
  volume={2007},
  pages={35},
  year={2007},
  organization={New York}
}

@inproceedings{gao2022kuairec,
  author = {Gao, Chongming and Li, Shijun and Lei, Wenqiang and Chen, Jiawei and Li, Biao and Jiang, Peng and He, Xiangnan and Mao, Jiaxin and Chua, Tat-Seng},
  title = {KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems},
  booktitle = {Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  series = {CIKM '22},
  location = {Atlanta, GA, USA},
  url = {https://doi.org/10.1145/3511808.3557220},
  doi = {10.1145/3511808.3557220},
  numpages = {11},
  year = {2022},
  pages = {540–550}
}
```

If things go well, you can run the following examples now.

---
## Examples to run the code

The following commands only give one example. For more details, please kindly refer to the paper and the definitions in the code files. 

- #### Movielens-100K

1. Train the the RL policy using Movielens-100K

    ```bash
    python main.py --dataset 'ml100k'
    ```

2. Train the the RL policy using Movielens-100K with Top-10

    ```bash
    python main.py --dataset 'ml100k' --topk 10
    ```

---
## A guide to reproduce the main results of the paper.

You can follow the guide to reproduce the main results of baselines, see [ItemPop](https://recbole.io/docs/user_guide/model/general/pop.html), [DIN](https://recbole.io/docs/user_guide/model/context/din.html), [MF-IPS and DICE](https://github.com/JingsenZhang/Recbole-Debias), [C2UCB](https://github.com/YunSeo00/combinatorial_MAB), [FCPO](https://github.com/TobyGE/FCPO).

The details of VirtualTaobao can be referred to [VirtualTaobao](https://github.com/eyounx/VirtualTaobao).
