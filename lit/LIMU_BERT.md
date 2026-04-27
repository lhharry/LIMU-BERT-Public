# LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications

**Authors:** Huatao Xu, Pengfei Zhou, Rui Tan, Mo Li (Nanyang Technological University, Singapore); Guobin Shen (Alibaba Group, China)

**Published in:** The 19th ACM Conference on Embedded Networked Sensor Systems (SenSys'21), November 15–17, 2021, Coimbra, Portugal

**DOI:** https://doi.org/10.1145/3485730.3485937

---

## Abstract

Deep learning greatly empowers Inertial Measurement Unit (IMU) sensors for various mobile sensing applications, including human activity recognition, human-computer interaction, localization and tracking, and many more. Most existing works require substantial amounts of well-curated labeled data to train IMU-based sensing models, which incurs high annotation and training costs. Compared with labeled data, unlabeled IMU data are abundant and easily accessible. In this work, we present LIMU-BERT, a novel representation learning model that can make use of unlabeled IMU data and extract generalized rather than task-specific features. LIMU-BERT adopts the principle of self-supervised training of the natural language model BERT to effectively capture temporal relations and feature distributions in IMU sensor measurements. However, the original BERT is not adaptive to mobile IMU data. By meticulously observing the characteristics of IMU sensors, we propose a series of techniques and accordingly adapt LIMU-BERT to IMU sensing tasks. The designed models are lightweight and easily deployable on mobile devices. With the representations learned via LIMU-BERT, task-specific models trained with limited labeled samples can achieve superior performances. We extensively evaluate LIMU-BERT with four open datasets. The results show that the LIMU-BERT enhanced models significantly outperform existing approaches in two typical IMU sensing applications.

## CCS Concepts

- Human-centered computing → Ubiquitous and mobile computing systems and tools
- Computing methodologies → Machine learning

## Keywords

IMU, Mobile Sensing, Representation Learning, BERT

---

## 1. Introduction

In recent years, the proliferation of embedded and mobile devices unveils the era of Artificial Intelligence of Things (AIoT). Particularly, wearable devices have played a critical role in a wide range of applications, including human activity recognition, human-computer interaction, localization and tracking, etc. Many of them highly rely on the data from Inertial Measurement Unit (IMU) sensors (i.e., accelerometer, gyroscope, and magnetometer), which are widely equipped in personal mobile devices, such as smartphones, smartwatches, and even smart earphones.

Due to the rapid development of deep learning, many works adopt deep neural networks to process IMU data. Compared with manual feature engineering, deep learning algorithms can extract more effective features and gain significant performance improvements in inference. Most existing works, however, rely heavily on supervised learning processes where substantial amounts of labeled IMU data are required to train sensing models. The requirement of large labeled data hinders their adoption in practice for two reasons. First, labeled IMU data are scarce because it is costly and time-consuming to collect sufficient labeled IMU samples in the real-world settings. Second, the diversity in mobile devices, usage patterns, and environments results in the need for labeled data with various combinations of phone models, users, and usage scenarios to attain generalizable models.

To address the challenge of labeled data scarcity, this paper proposes a representation learning model that can leverage massive unlabeled data to extract general features through self-supervised training technique. After the representations are learned, multiple task-specific inference models can thus be trained with a small amount of labeled IMU samples. The key rationale is to learn the generalizable representations from the abundant unlabeled IMU data instead of scarce labeled data.

To design such a representation learning model, we first answer the following basic question: **what general features are desired from the IMU data?** After scrutinizing the characteristics of IMU data, we focus on two types of features: distributions of individual measurements of IMU sensors, and temporal relations in continuous measurements.

Our model design is thereafter guided by answering: **how to extract those general features or representations from unlabeled IMU data?** Inspired by the emerging self-supervised techniques in natural language processing, we borrow the key framework of BERT to process unlabeled IMU data and accordingly extract general features. However, intended for natural language data processing, original BERT lacks methodology in processing IMU data, e.g., multi-modality problem of various IMU sensor readings. This paper thus devises a variety of techniques including data fusion and normalization, effective training method, structure optimization, and embeds them into the BERT framework for improved efficacy and efficiency in IMU sensing applications.

We name our model **LIMU-BERT**, which stands for a lite BERT-like self-supervised representation learning model for mobile IMU data.

### Contributions

- This paper devises a self-supervised approach to learn general representations from unlabeled IMU data. Based on learned representations, task-specific models can be trained with few labeled samples, which substantially reduces the supervised training overhead with labeled data.
- This paper proposes a series of adaptations and enhancements around BERT to best work with IMU data in mobile sensing applications. The proposed LIMU-BERT is lightweight, which can be accommodated in mobile devices.
- A prototype system is developed and experimentally evaluated. Extensive evaluation results show the effectiveness of LIMU-BERT in learning generalizable data representations. The codes are made publicly available at https://github.com/dapowan/LIMU-BERT-Public.

---

## 2. Preliminaries

### 2.1 Representation Learning

Representation learning techniques aim to extract the representations or general features from raw data. Traditional methods rely on domain expertise or prior knowledge to engineer features. Recent studies show that automatic representation learning with deep neural network is effective when provided large labeled data. It is however costly and time-consuming to gather a dataset with clean labels. To make use of large amounts of unlabeled data, more advanced models have been proposed to extract representations from unlabeled images, video, or texts. The process of learning representations from unlabeled data is called **self-supervised learning**.

**BERT (Bidirectional Encoder Representations from Transformers)** is one effective self-supervised learning model for Natural Language Processing (NLP). BERT features two self-supervised tasks:

- **Masked Language Model (MLM):** Randomly masks parts of the input text and trains the model to predict the original identity numbers of the masked words.
- **Next Sentence Prediction (NSP):** Requires the model to determine whether two given sentences are subsequent or not.

Through MLM and NSP, BERT can learn contextual relations in text data and accordingly generate effective embeddings for each word.

### 2.2 Uniqueness of IMU Sensing

By examining four sets of IMU readings of different human activities with different device placements, we obtain the following key observations:

**Fusion matters.** Cross referencing of multiple sensors can provide more information and improves the overall performance. The representation learning model should support the data fusion with multiple IMU sensors, which is not the design objective of the original BERT for NLP.

**Distribution matters.** By comparing individual measurements of accelerometer and gyroscope in different activities, we find their ranges greatly vary. For example, the gyroscope readings are within (−5, 5) when the user walks and are distributed between -15 and 15 when the user runs. The distribution of IMU readings contains rich information that LIMU-BERT should capture. Any transformation that may destruct the distribution information of raw IMU data should not be applied before feeding them into neural network.

**Context matters.** Walk and run exhibit obvious periodical patterns on the IMU data, which is a reliable feature that distinguishes them from standing still. Temporal relations also play an important role in representation learning for IMU data, which will likely benefit from a BERT-like design.

**Efficiency matters.** LIMU-BERT targets processing IMU data collected from mobile devices in real time. The base model of BERT has about 110 million parameters, which is too heavy for mobile devices. A lightweight and efficient design is needed.

### 2.3 Potential Applications

For influential applications of IMU sensors (HAR, indoor localization, device attitude estimation, etc.), LIMU-BERT aims at extracting general features to achieve superior performance with limited labeled samples. This paper demonstrates the efficacy of LIMU-BERT using:

- **Human Activity Recognition (HAR)**
- **Device Placement Classification (DPC)**

---

## 3. Design

### 3.1 Overview

The framework consists of self-supervised and supervised learning phases. There are three major components:

1. **LIMU-BERT** — takes the unlabeled IMU data as input and outputs high-level representations or features.
2. **Decoder** — reconstructs the unlabeled data based on the learned features.
3. **Classifier** — trained with a small amount of labeled representations to accomplish a task-specific application.

**Self-supervised learning.** We mask partial readings of unlabeled samples and feed them into LIMU-BERT. LIMU-BERT and the decoder jointly predict the original values of masked readings by learning the temporal relations among IMU data.

**Supervised learning.** We transfer the LIMU-BERT model and connect it with a classifier. All parameters of the LIMU-BERT are frozen and only the classifier is trained with limited labeled representations.

### 3.2 Fusion and Normalization

The sensor measurements need to be properly normalized before being fed into LIMU-BERT. We design a simple but effective normalization method on accelerometer and magnetometer readings:

$$acc_i = \frac{acc_i}{9.8 \, m/s^2}, \quad mag_i = \frac{\alpha \cdot mag_i}{\sqrt{\sum mag_i^2}}, \quad i \in \{x, y, z\}$$

where α is a weight scaling the range of the magnetometer readings (set to 2 in LIMU-BERT). We keep the distribution of gyroscope readings because they are naturally small values.

To extend the dimension of features and fuse IMU sensors, we project the normalized sensor data X into a higher space:

$$I = \text{Proj}(X) = W \times X$$

W is a matrix of size H_dim × S_dim, where H_dim is the hidden dimension larger than S_dim.

LIMU-BERT then leverages **Layer Normalization** to normalize the fused features:

$$I'_{ij} = \text{LayerNorm}(I) = \frac{I_{ij} - E(I_{\cdot j})}{\sqrt{Var(I_{\cdot j}) + \epsilon}} \cdot \gamma + \beta$$

### 3.3 Learning Representations

After analyzing the two self-supervised tasks (MLM and NSP) in BERT, we find the NSP task is not suitable for IMU data due to the frequent transitions of human daily activities. The MLM task is beneficial to extracting our target features from IMU data.

We implement a **Span Masking mechanism**, which samples the length of masked subsequence from a geometric distribution Geo(p) clipped at l_max:

$$P(l = k) = (1 - p)^{k-1} p, \quad s.t. \, l \in [1, l_{max}]$$

The probability of success p is set to 0.2 and l_max is set to 10.

#### Algorithm 1: Span Mask Algorithm

```
Input: IMU sequence X, sequence length L, probability of success p, 
       masked ratio p_r, mask probability P_m
Output: Masked IMU sequence X, masked position set I

1: M_max = L × p_m, m = 0, I = ∅
2: sample p_m from U[0, 1)
3: while m < M_max do
4:     sample s from U[0, L)
5:     if s ∉ I then
6:         sample l from Geo(p)
7:         l = min(l, M_max - m), e = min(s + l, L)
8:         for j = s to e do
9:             I = I ∪ {j}, m = m + 1
10:            if p_m < P_m then
11:                X_·j = 0
12:            end if
13:        end for
14:    end if
15: end while
```

The masked ratio p_r and masking probability P_m are set to 0.15 and 0.8, respectively.

### 3.4 Lightweight Model

Three customizations make LIMU-BERT lightweight:

1. **Smaller sampling rate (20 Hz)** compared with existing works, decreasing the length of input IMU sequences.
2. **Cross-layer parameter sharing mechanism** — only the parameters in the first encoder layer are trained and shared with other layers.
3. **Regression task formulation** — IMU data reconstruction is treated as a regression task rather than a classification task.

### 3.5 LIMU-BERT Design

**LIMU-BERT.** The objective is:

$$E = f_{enc}(X^u)$$

where E is an H_dim × L matrix. After positional encoding:

$$H^{\{0\}}_{\cdot j} = \text{LayerNorm}(I'_{\cdot j} + PE(j))$$

The attention-enteric block contains three residual blocks:

$$H^{\{r\}} = \text{LayerNorm}(\text{FeedForward}(P^{\{r-1\}}) + P^{\{r-1\}})$$

$$P^{\{r-1\}} = \text{LayerNorm}(\text{Proj}(A^{\{r-1\}}) + A^{\{r-1\}})$$

$$A^{\{r-1\}} = \text{LayerNorm}(\text{MultiAttn}(H^{\{r-1\}}) + H^{\{r-1\}})$$

In LIMU-BERT, R_num and H_dim are set to 4. L is set to 120 under a sampling rate of 20Hz.

**Decoder.**

$$\hat{X}^u = f_{dec}(E)$$

$$\hat{X}^u = \text{LayerNorm}(\text{Pred}(D))$$

$$D = \text{Proj}(\text{GELU}(E))$$

**Training loss:**

$$loss = \frac{1}{|X^u|} \sum_{i=1}^{|X^u|} \text{MSE}(\text{Select}(X^u_{[i]}), \text{Select}(\hat{X}^u_{[i]}))$$

### 3.6 Task-specific Classifier Design

The lightweight GRU classifier contains three stacked GRU layers with hidden sizes of 20, 20, and 10, respectively. The dropout rate is 0.5. Two fully-connected layers with 10 hidden units are constructed before the softmax layer.

Alternative classifiers include CNN-based and Multi-head Attention-based variants.

---

## 4. Evaluation

### 4.1 Methodology

#### 4.1.1 Datasets

| Dataset | Sensor | Activity | User | Placement | Sample |
|---------|--------|----------|------|-----------|--------|
| HHAR | A,G | 6 | 9 | - | 9166 |
| UCI | A,G | 6 | 30 | - | 2088 |
| MotionSense | A,G | 6 | 24 | - | 4534 |
| Shoaib | A,G,M | 7 | 10 | 5 | 10500 |

(A=accelerometer, G=gyroscope, M=magnetometer)

#### 4.1.2 Preprocessing

For all datasets, we down-sample to 20 Hz and slice the continuous IMU data into windows of 120 measurements. Each dataset is randomly divided into training (80%), validation (10%), and test (10%) sets. The training set is further divided into 1% labeled set and 99% unlabeled set.

#### 4.1.3 Models in Comparison

- **LIMU-GRU** — classification model implemented based on our framework.
- **DCNN** — deep CNN-based model.
- **DeepSense** — applies Fourier Transform to raw IMU data.
- **TPN** — multi-task temporal CNN trained to recognize transformations on input data.
- **R-GRU** — baseline model that directly applies GRU classifier on raw IMU data.

#### 4.1.4 Implementation

Implemented with Python and PyTorch. Trained on a server with 4 NVIDIA GEFORCE 2080Ti GPUs. Learning rate is 0.001 and batch size is 128. LIMU-BERT and TPN are pre-trained for 3,200 epochs; classifiers and baselines are trained for 700 epochs.

#### 4.1.5 Application and Metrics

Two applications: HAR and DPC. Metrics: accuracy and macro F-score.

### 4.2 Evaluation of Human Activity Recognition

#### 4.2.1 Overall Performances (1% labeled data)

| Dataset | HHAR | | UCI | | MotionSense | | Shoaib | | Average | |
|---------|------|------|-----|-----|------|------|------|------|---------|---------|
| Metric | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 |
| DCNN | 0.760 | 0.736 | 0.649 | 0.625 | 0.721 | 0.637 | 0.715 | 0.718 | 0.711 | 0.679 |
| DeepSense | 0.715 | 0.688 | 0.576 | 0.544 | 0.722 | 0.650 | 0.682 | 0.683 | 0.674 | 0.641 |
| R-GRU | 0.849 | 0.832 | 0.760 | 0.741 | 0.846 | 0.806 | 0.785 | 0.787 | 0.810 | 0.792 |
| TPN | 0.250 | 0.151 | 0.208 | 0.068 | 0.084 | 0.026 | 0.163 | 0.040 | 0.176 | 0.071 |
| **LIMU-GRU** | **0.964** | **0.962** | **0.924** | **0.923** | **0.927** | **0.899** | **0.900** | **0.899** | **0.929** | **0.921** |

LIMU-GRU outperforms baseline models by at least 10% in all cases.

#### 4.2.2 Varying Labeling Rate

LIMU-GRU consistently outperforms the baselines when labeling rate varies from 0.2% to 10%. The performance gap is higher when labeling rate is smaller. At 0.2% labeling rate, LIMU-GRU achieves accuracies of 0.863, 0.875, and 0.855 on HHAR, UCI, and MotionSense, while all other models achieve below 0.5.

#### 4.2.3 Varying Sequence Length

In all sequence length settings (20, 40, 60, 120), LIMU-BERT performs the best. Longer IMU sample sequences do not always lead to higher performance.

### 4.3 Evaluation of Device Placement Classification

| Labeling rate | 0.2% | | 0.5% | | 1% | | 2% | | 5% | | 10% | | Average | |
|---------------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| Metric | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 |
| DCNN | 0.486 | 0.484 | 0.561 | 0.564 | 0.657 | 0.652 | 0.781 | 0.778 | 0.842 | 0.839 | 0.900 | 0.897 | 0.705 | 0.702 |
| DeepSense | 0.463 | 0.457 | 0.537 | 0.529 | 0.600 | 0.594 | 0.669 | 0.663 | 0.765 | 0.761 | 0.819 | 0.815 | 0.642 | 0.637 |
| R-GRU | 0.613 | 0.589 | 0.719 | 0.716 | 0.832 | 0.830 | 0.901 | 0.901 | 0.941 | 0.941 | 0.964 | 0.964 | 0.828 | 0.824 |
| TPN | 0.321 | 0.184 | 0.421 | 0.319 | 0.404 | 0.266 | 0.488 | 0.391 | 0.336 | 0.216 | 0.391 | 0.257 | 0.394 | 0.272 |
| **LIMU-GRU** | **0.753** | **0.746** | **0.886** | **0.885** | **0.920** | **0.921** | **0.948** | **0.949** | **0.969** | **0.970** | **0.984** | **0.984** | **0.910** | **0.909** |

### 4.4 Micro-benchmark

#### 4.4.1 Representation Visualization

Using t-SNE, the learned representations show high clustering effect for samples belonging to the same activity class. Representations of dynamic activities (walking, jogging, upstairs, downstairs) are likely to be close.

#### 4.4.2 Varying Classifier

| Application | HAR | | HAR | | HAR | | HAR | | DPC | | Average | |
|-------------|-----|------|-----|------|-----|------|-----|------|-----|------|---------|---------|
| Dataset | HHAR | | UCI | | MotionSense | | Shoaib | | Shoaib | | - | |
| Metric | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 |
| LIMU-CNN | 0.952 | 0.946 | 0.883 | 0.882 | 0.895 | 0.858 | 0.849 | 0.850 | 0.884 | 0.884 | 0.893 | 0.884 |
| LIMU-ATTN | 0.928 | 0.923 | 0.915 | 0.913 | 0.909 | 0.874 | 0.809 | 0.810 | 0.812 | 0.811 | 0.875 | 0.866 |
| LIMU-LSTM | 0.953 | 0.949 | 0.913 | 0.915 | 0.913 | 0.880 | 0.890 | 0.891 | **0.921** | **0.921** | 0.918 | 0.911 |
| **LIMU-GRU** | **0.964** | **0.962** | **0.924** | **0.923** | **0.927** | **0.899** | **0.900** | **0.899** | 0.920 | 0.921 | **0.927** | **0.921** |

#### 4.4.3 Varying Sensors

The performance gain introduced by extra gyroscope readings is significant. Magnetometer readings do not bring much benefit but do not degrade performance.

#### 4.4.4 Varying Normalization Method

The normalization method adopted in LIMU-BERT outperforms using raw data by 5.78% on average. Mean-variance normalization causes severe performance degradation.

#### 4.4.5 Varying Masking Approach

Span masking obtains higher average accuracy than single masking. The setting of p = 0.2 achieves the best overall performances.

#### 4.4.6 Varying Representation Dimension

H_dim is set to 72, which achieves best performances. F1-scores increase when H_dim is under 72, but decrease if H_dim increases to 144 due to over-fitting.

#### 4.4.7 Varying Dataset (Cross-Dataset Performance)

| Source Dataset | HHAR HAR | UCI HAR | Motion HAR | Shoaib HAR | Shoaib DPC | Avg. |
|----------------|----------|---------|------------|------------|------------|------|
| HHAR | 0.964 | 0.862 | 0.872 | 0.845 | 0.832 | 0.875 |
| UCI | 0.865 | 0.924 | 0.879 | 0.843 | 0.820 | 0.866 |
| Motion | 0.883 | 0.879 | 0.927 | 0.847 | 0.852 | 0.878 |
| Shoaib | 0.879 | 0.847 | 0.869 | 0.900 | 0.925 | 0.884 |
| **Merged** | **0.905** | **0.932** | **0.901** | **0.895** | **0.883** | **0.903** |

#### 4.4.8 Computation Overhead

| Model | Parameters | Size | Train Time | Infer. Time |
|-------|-----------|------|------------|-------------|
| DCNN | 17 K | 77 KB | 4 ms | 6 ms |
| DeepSense | 13 K | 73 KB | 8 ms | 6 ms |
| TPN | 105 K | 501 KB | 38+2 ms | 6 ms |
| R-GRU | 5 K | 24 KB | 4 ms | 18 ms |
| LIMU-BERT* | 189 K | 766 KB | 36 ms | 18 ms |
| LIMU-BERT | 62 K | 255 KB | 27 ms | 14 ms |
| LIMU-GRU | 9 K | 39 KB | 6 ms | 18 ms |

LIMU-BERT* represents LIMU-BERT without cross-layer parameter sharing and the decoder. Inference time measured on Samsung Galaxy S8.

---

## 5. Related Work

Applying deep learning techniques with IMU sensors in mobile devices facilitates many ubiquitous applications, such as human activity recognition, human-computer interaction, user authentication, and indoor tracking. However, most models are trained with a large amount of labeled samples requiring great manual labeling efforts. MetaSense employs meta-learning but still needs a large labeled dataset. EI proposes feature extraction from labeled and unlabeled data but needs multiple types of labels.

Self-supervised learning approaches have been widely studied to reduce dependence on labeled data. TPN is the only existing work that borrowed the idea of self-supervised learning and applied it to IMU sensing. However, TPN is designed to process only accelerometer readings and is limited in fully unleashing the potential of multi-modality IMU sensors.

Different from TPN, LIMU-BERT can handle multiple sensor data thanks to its special design in normalization and fusion. LIMU-BERT targets two types of features (distributions of individual measurements and temporal relations) and learns them by adaptive MLM self-supervised task.

---

## 6. Discussion and Future Work

**Model transferability.** Performance slightly degrades when transferring across datasets due to diversity in devices, placements, users, and environments. Future improvements may use techniques like denoised autoencoder or data augmentation.

**Irrelevant event detection.** As a generative self-supervised model, LIMU-BERT is sensitive to rare samples and may fail to extract effective features from them.

**User privacy.** Sensor data uploaded to cloud may cause privacy issues. Federated learning may be introduced for protecting user privacy.

Other future works include investigating how representations learned by LIMU-BERT may facilitate other mobile applications, e.g., indoor localization or device orientation estimation.

---

## 7. Conclusion

In this paper, we present a lite BERT-like representation learning model for mobile IMU sensor data, which makes use of unlabeled data and accordingly extracts generalizable features instead of task-specific features. Extensive experimental evaluation demonstrates the learned representations by LIMU-BERT can boost the performances of down-stream models significantly with few labeled samples. With LIMU-BERT, the labeling efforts in real IMU-based sensing applications can be greatly reduced.

---

## Acknowledgments

This research is supported by the National Research Foundation, Singapore under its Industry Alignment Fund – Pre-positioning (IAF-PP) Funding Initiative, Alibaba Group through Alibaba Innovative Research (AIR) Program and Alibaba-NTU Singapore Joint Research Institute (JRI), and NTU CoE SUG.

---

## References

1. Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. Layer Normalization. arXiv:1607.06450 [stat.ML]
2. Cheng Bo, Lan Zhang, Xiang-Yang Li, Qiuyuan Huang, and Yu Wang. 2013. Silentsense: silent user identification via touch and movement behavioral biometrics. In Proceedings of the 19th annual international conference on Mobile computing & networking. 187–190.
3. Wenqiang Chen, Lin Chen, Yandao Huang, Xinyu Zhang, Lu Wang, Rukhsana Ruby, and Kaishun Wu. 2019. Taprint: Secure text input for commodity smart wristbands. In The 25th Annual International Conference on Mobile Computing and Networking. 1–16.
4. Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.
5. Zhi-An Deng, Guofeng Wang, Ying Hu, and Di Wu. 2015. Heading estimation for indoor pedestrian navigation using a smartphone in the pocket. Sensors 15, 9 (2015), 21518–21536.
6. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
7. Basura Fernando, Hakan Bilen, Efstratios Gavves, and Stephen Gould. 2017. Self-supervised video representation learning with odd-one-out networks. In Proceedings of the IEEE conference on computer vision and pattern recognition. 3636–3645.
8. Taesik Gong, Yeonsu Kim, Jinwoo Shin, and Sung-Ju Lee. 2019. Metasense: few-shot adaptation to untrained conditions in deep mobile sensing. In Proceedings of the 17th Conference on Embedded Networked Sensor Systems. 110–123.
9. Dan Hendrycks and Kevin Gimpel. 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.
10. Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation 9, 8 (1997), 1735–1780.
11. Nathalie Japkowicz and Shaju Stephen. 2002. The class imbalance problem: A systematic study. Intelligent data analysis 6, 5 (2002), 429–449.
12. Wenjun Jiang et al. 2018. Towards environment independent device free human activity recognition. In MobiCom. 289–304.
13. Wenchao Jiang and Zhaozheng Yin. 2015. Human activity recognition using wearable sensors by deep convolutional neural networks. In ACM MM. 1307–1310.
14. Yonghang Jiang, Zhenjiang Li, and Jianping Wang. 2018. Ptrack: Enhancing the applicability of pedestrian tracking with wearables. IEEE TMC 18, 2 (2018), 431–443.
15. Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S Weld, Luke Zettlemoyer, and Omer Levy. 2020. Spanbert: Improving pre-training by representing and predicting spans. TACL 8 (2020), 64–77.
16. Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
17. Jakub Konečný et al. 2016. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492.
18. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. Imagenet classification with deep convolutional neural networks. NeurIPS 25 (2012), 1097–1105.
19. Zhenzhong Lan et al. 2019. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942.
20. Hsin-Ying Lee, Jia-Bin Huang, Maneesh Singh, and Ming-Hsuan Yang. 2017. Unsupervised representation learning by sorting sequences. In ICCV. 667–676.
21. Xinyu Li, Yanyi Zhang, Ivan Marsic, Aleksandra Sarcevic, and Randall S Burd. 2016. Deep learning for rfid-based activity recognition. In SenSys. 164–175.
22. Jian Liu et al. 2019. Wireless sensing for human activity: A survey. IEEE Communications Surveys & Tutorials 22, 3 (2019), 1629–1645.
23. Shengzhong Liu et al. 2020. GIobalFusion: A Global Attentional Deep Learning Framework for Multisensor Information Fusion. IMWUT 4, 1 (2020), 1–27.
24. Yang Liu, Zhenjiang Li, Zhidan Liu, and Kaishun Wu. 2019. Real-time arm skeleton tracking and gesture inference tolerant to missing wearable sensors. In MobiSys. 287–299.
25. Mohammad Malekzadeh, Richard G Clegg, Andrea Cavallaro, and Hamed Haddadi. 2019. Mobile sensor data anonymization. In IoTDI. 49–58.
26. Brendan McMahan et al. 2017. Communication-efficient learning of deep networks from decentralized data. In AISTATS. 1273–1282.
27. Ishan Misra, C Lawrence Zitnick, and Martial Hebert. 2016. Shuffle and learn: unsupervised learning using temporal order verification. In ECCV. 527–544.
28. Adam Paszke et al. 2017. Automatic differentiation in pytorch.
29. Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. 2017. Curiosity-driven exploration by self-supervised prediction. In ICML. 2778–2787.
30. Ronald Poppe. 2010. A survey on vision-based human action recognition. Image and vision computing 28, 6 (2010), 976–990.
31. Zhen Qin et al. 2019. Learning-aided user identification using smartphone sensors for smart homes. IEEE IoT Journal 6, 5 (2019), 7760–7772.
32. Jorge-L Reyes-Ortiz et al. 2016. Transition-aware human activity recognition using smartphones. Neurocomputing 171 (2016), 754–767.
33. Aaqib Saeed, Tanir Ozcelebi, and Johan Lukkien. 2019. Multi-task self-supervised learning for human activity detection. IMWUT 3, 2 (2019), 1–30.
34. Sheng Shen, Mahanth Gowda, and Romit Roy Choudhury. 2018. Closing the gaps in inertial motion tracking. In MobiCom. 429–444.
35. Sheng Shen, He Wang, and Romit Roy Choudhury. 2016. I am a smartwatch and i can track my user's arm. In MobiSys. 85–96.
36. Muhammad Shoaib, Stephan Bosch, Ozlem Durmaz Incel, Hans Scholten, and Paul JM Havinga. 2014. Fusion of smartphone motion sensors for physical activity recognition. Sensors 14, 6 (2014), 10146–10176.
37. Connor Shorten and Taghi M Khoshgoftaar. 2019. A survey on image data augmentation for deep learning. Journal of Big Data 6, 1 (2019), 1–48.
38. Yuanchao Shu et al. 2015. Magicol: Indoor localization using pervasive magnetic field and opportunistic WiFi sensing. IEEE JSAC 33, 7 (2015), 1443–1457.
39. Yuanchao Shu, Kang G Shin, Tian He, and Jiming Chen. 2015. Last-mile navigation using smartphones. In MobiCom. 512–524.
40. Allan Stisen et al. 2015. Smart devices are different: Assessing and mitigating mobile sensing heterogeneities for activity recognition. In SenSys. 127–140.
41. Scott Sun, Dennis Melamed, and Kris Kitani. 2021. IDOL: Inertial Deep Orientation-Estimation and Localization. In AAAI 35. 6128–6137.
42. Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. JMLR 9, 11 (2008).
43. Ashish Vaswani et al. 2017. Attention is all you need. arXiv preprint arXiv:1706.03762.
44. Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. Extracting and composing robust features with denoising autoencoders. In ICML. 1096–1103.
45. Hao Wang et al. 2016. RT-Fall: A real-time and contactless fall detection system with commodity WiFi devices. IEEE TMC 16, 2 (2016), 511–526.
46. Wei Wang, Alex X Liu, Muhammad Shahzad, Kang Ling, and Sanglu Lu. 2015. Understanding and modeling of wifi signal based human activity recognition. In MobiCom. 65–76.
47. Tianzhang Xing et al. 2020. DWatch: A Reliable and Low-Power Drowsiness Detection System for Drivers Based on Mobile Devices. ACM TOSN 16, 4, Article 37 (Sept. 2020), 22 pages.
48. Xiangyu Xu et al. 2020. TouchPass: towards behavior-irrelevant on-touch user authentication on smartphones leveraging vibrations. In MobiCom. 1–13.
49. Jianbo Yang, Minh Nhut Nguyen, Phyo Phyo San, Xiaoli Li, and Shonali Krishnaswamy. 2015. Deep convolutional neural networks on multichannel time series for human activity recognition. In IJCAI 15. 3995–4001.
50. Zhijian Yang, Yu-Lin Wei, Sheng Shen, and Romit Roy Choudhury. 2020. Ear-AR: indoor acoustic augmented reality on earphones. In MobiCom. 1–14.
51. Shuochao Yao, Shaohan Hu, Yiran Zhao, Aston Zhang, and Tarek Abdelzaher. 2017. Deepsense: A unified deep learning framework for time-series mobile sensing data processing. In WWW. 351–360.
52. Yinggang Yu, Dong Wang, Run Zhao, and Qian Zhang. 2019. RFID based real-time recognition of ongoing gesture with adversarial learning. In SenSys. 298–310.
53. Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. 2019. S4l: Self-supervised semi-supervised learning. In ICCV. 1476–1485.
54. Yi Zhang et al. 2021. XGest: Enabling Cross-Label Gesture Recognition with RF Signals. ACM TOSN 17, 4, Article 37 (Sept. 2021), 23 pages.
55. Yi Zhao et al. 2020. Urban Scale Trade Area Characterization for Commercial Districts with Cellular Footprints. ACM TOSN 16, 4, Article 42 (Sept. 2020), 20 pages.
56. Yuanqing Zheng et al. 2017. Travi-navi: Self-deployable indoor navigation system. IEEE/ACM ToN 25, 5 (2017), 2655–2669.
57. Yue Zheng et al. 2019. Zero-effort cross-domain gesture recognition with Wi-Fi. In MobiSys. 313–325.
58. Han Zhou, Yi Gao, Xinyi Song, Wenxin Liu, and Wei Dong. 2019. LimbMotion: Decimeter-level Limb Tracking for Wearable-based Human-Computer Interaction. IMWUT 3, 4 (2019), 1–24.
59. Pengfei Zhou, Mo Li, and Guobin Shen. 2014. Use it free: Instantly knowing your phone attitude. In MobiCom. 605–616.
60. Pengfei Zhou, Yuanqing Zheng, and Mo Li. 2012. How long to wait? Predicting bus arrival time with mobile phone based participatory sensing. In MobiSys. 379–392.
