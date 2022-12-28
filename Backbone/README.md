# Backbone Classification Models

All backbone models are trained on the ImageNet dataset with training setups following [1].
Specifically, I use the following:

- Regularization: Batch norm layers after **every** convolution layer
    - Momentum: 0.9 (the original paper calls for 0.99)
    - Weight decay: 1e-5
- Optimizer: RMSProp
    - Decay: 0.9
    - Momentum: 0.9
- Learning Rate: 0.256
    - Decay: 0.97 every 2 epochs (the original paper calls for 2.4)


<br>

Performance for the backbones can be summarized as follows. Note that each metric is the **validation performance** of the model, as I have not submitted the results to the official evaluation server.

| **Model**  | **Acc@1** | **Acc@5** | **Loss** | **Weights**   |
|------------|-----------|-----------|----------|------------|
| Darknet53  | 77.34     | 93.49     | 0.895    | [link](   )|
| Darknet(24)| 73.82     | 91.59     | 1.057    | [link](   )|
| Darknet19  | 71.19     | 90.30     | 1.162    | [link](   )|

---

[1]: M. Tan, B. Chen, R. Pang, V. Vasudevan, M. Sandler, A. Howard, Q. Le. [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf). *CVPR*, 2019.