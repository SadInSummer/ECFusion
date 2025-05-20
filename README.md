# ECFusion
This is code of "ECFusion: Edge-guided Cross-scale Fusion Based on Generative Adversarial Learning for Multi-modal Medical Image Fusion"(TSH'25)


Just prepared the data and the filefloders, and prepared the vgg pre-trained (it will be download when you python this train.py and test.py)
![image](https://github.com/user-attachments/assets/4b00d472-c57f-4538-916a-dd08aaa354c4)

and then 
python the train.py and test.py.

and the prepared model is avaliable  at :https://drive.google.com/file/d/1vW3w_jaiyMXNiahf_W-uGSA61i4oWkeg/view?usp=drive_link



@article{Wei2025, 
author = {Xinjian Wei and Qiu Yu and Xiaoxuan Xu and Jing Xu and Bin Wang},
title = {ECFusion: Edge-guided Cross-scale Fusion Based on Generative Adversarial Learning for Multi-modal Medical Image Fusion},
year = {2025},
journal = {Tsinghua Science and Technology},
keywords = {generative adversarial networks, transformer, multi-modal medical image fusion, edge information, multi-modal interaction},
url = {https://www.sciopen.com/article/10.26599/TST.2025.9010054},
doi = {10.26599/TST.2025.9010054},
abstract = {Multi-modal medical image fusion effectively integrates structural and functional information, compensating the limitations inherent in single-modality imaging. Current methods prioritize the salient characteristics of organs and tissues, neglecting the significance of edge features, consequently, the fused results usually exhibit roughness and blurred edge details. To deal with this issue, we propose a novel Edge-guided Cross-scale Fusion based on generative adversarial learning, namely ECFusion. In details, we present a novel edge-guided cross-scale fusion mechanism that intelligently guides the network to focus on the edge information, while emphasizing the salient region features for comprehensive representation. Besides, we design a new commute-Entropy Region Mutual Information loss, ensuring the automatic interaction and balance by exchanging the information between different modalities. Extensive experiments reveal our ECFusion outperforms the state-ofthe- art methods on various evaluation metrics, demonstrating the high accuracy and efficiency of our proposed on different fusion tasks.}
}


Thanks for the TGFuse.
