# FedPalm

This repository is a PyTorch implementation of FedPalm (FedPalm: A General Federated Learning Framework for Closed- and Open-Set Palmprint Verification). The paper can be accessed at [this link](https://arxiv.org/abs/2503.04837). 

#### Abstract
Current deep learning (DL)-based palmprint verification models rely on centralized training with large datasets, which raises significant privacy concerns due to the sensitive and immutable nature of biometric data. Federated learning~(FL), a privacy-preserving distributed learning paradigm, offers a compelling alternative by enabling collaborative model training without the need for data sharing. However, FL-based palmprint verification faces critical challenges, including data heterogeneity from diverse identities and the absence of standardized evaluation benchmarks. This paper addresses these gaps by establishing a comprehensive benchmark for FL-based palmprint verification, which explicitly defines and evaluates two practical scenarios: closed-set and open-set verification. We propose FedPalm, a unified FL framework that balances local adaptability with global generalization. Each client trains a personalized textural expert tailored to local data and collaboratively contributes to a shared global textural expert for extracting generalized features. To further enhance verification performance, we introduce a Textural Expert Interaction Module that dynamically routes textural features among experts to generate refined side textural features. Learnable parameters are employed to model relationships between original and side features, fostering cross-texture-expert interaction and improving feature discrimination. Extensive experiments validate the effectiveness of FedPalm, demonstrating robust performance across both scenarios and providing a promising foundation for advancing FL-based palmprint verification research. 

#### Requirements

If you have already tried our previous works [CCNet](https://github.com/Zi-YuanYang/CCNet) or [CO3Net](https://github.com/Zi-YuanYang/CO3Net), you can skip this step.

Our codes were implemented by ```PyTorch 1.10``` and ```11.3``` CUDA version. Please first install necessary packages as follows:

```
pip install requirements.txt
```

#### Data Preparation
We aim to build a benchmark of FL in palmprint verification. The data distribution can be accessed at ``./data/`` (including PolyU, Tongji, and IITD). If you wanna try our method in other datasets, you need to generate training and testing texts as follows:

```
python ./data/genText.py
```

#### Training
After you prepare the training and testing texts, then you can directly run our training code. 

For PolyU:
```
python Train_FL_FedPalm.py --id_num 378 --train_set_file ./data/train_PolyU.txt --val_set_file ./data/val_PolyU.txt --gallery_set_file ./data/test_gallery_PolyU.txt  --query_set_file ./data/test_query_PolyU.txt --com 200
```

For Tongji:
```
python Train_FL_FedPalm.py --id_num 600 --train_set_file ./data/train_gallery_Tongji.txt --val_set_file ./data/train_query_Tongji.txt --gallery_set_file ./data/test_gallery_Tongji.txt  --query_set_file ./data/test_query_Tongji.txt --com 200
```

For IITD:
```
python Train_FL_FedPalm.py --id_num 460 --train_set_file ./data/train_IITD.txt --val_set_file ./data/val_IITD.txt --gallery_set_file ./data/test_gallery_IITD.txt  --query_set_file ./data/test_query_IITD.txt --com 100
```

#### Contact
If you have any question or suggestion to our work, please feel free to contact me. My email is cziyuanyang@gmail.com.

#### Citation
If our work is valuable to you, please cite our work:
```
@article{yang2025fedpalm,
  title={FedPalm: A General Federated Learning Framework for Closed-and Open-Set Palmprint Verification},
  author={Yang, Ziyuan and Chen, Yingyu and Gao, Chengrui and Teoh, Andrew Beng Jin and Zhang, Bob and Zhang, Yi},
  journal={arXiv preprint arXiv:2503.04837},
  year={2025}
}
```
