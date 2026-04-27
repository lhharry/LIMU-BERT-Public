# Experience Paper: Adopting Activity Recognition in On-demand Food Delivery Business

**Authors:** Huatao Xu¹*, Yan Zhang²*, Wei Gao², Guobin Shen³, Mo Li¹

¹ Hong Kong University of Science and Technology
² Rajax Network Technology (ele.me)
³ Hong Kong University of Science and Technology (Guangzhou)

*Huatao Xu and Yan Zhang contributed equally to the paper. Mo Li is the corresponding author.*

**Published in:** ACM MOBICOM '25: Proceedings of the 31st Annual International Conference on Mobile Computing and Networking, November 4–8, 2025, Hong Kong, China

**DOI:** https://doi.org/10.1145/3680207.3765261

---

## Abstract

This paper presents the first nationwide deployment of human activity recognition (HAR) technology in the on-demand food delivery industry. We successfully adapted the state-of-the-art LIMU-BERT foundation model to the delivery platform. Spanning three phases over two years, the deployment progresses from a feasibility study in Yangzhou City to nationwide adoption involving 500,000 couriers across 367 cities in China. The adoption enables a series of downstream applications, and large-scale tests demonstrate its significant operational and economic benefits, showcasing the transformative potential of HAR technology in real-world applications. Additionally, we share lessons learned from this deployment and open-source our LIMU-BERT pretrained with millions of hours of sensor data.

## CCS Concepts

- Human-centered computing → Ubiquitous and mobile computing systems and tools
- Computing methodologies → Machine learning approaches

## Keywords

Human Activity Recognition, Nationwide Deployment, On-demand Food Delivery, Business Adoption.

---

## 1. Introduction

The rapid advancement of Inertial Measurement Units (IMUs) has revolutionized various domains, ranging from robotics and automotive systems to consumer electronics such as smartphones, smartwatches, and earphones. These compact sensors serve as a cornerstone for a wide array of ubiquitous applications, including user authentication and motion tracking. Among these applications, Human Activity Recognition (HAR) has emerged as a pivotal technology that garnered significant attention in research.

However, there is no documented experience in the commercial adoption of this technology in large-scale scenarios. This paper addresses this gap by presenting our nationwide experience in applying the activity recognition model to the on-demand food delivery industry, which to our best knowledge, is the first of its kind. A typical delivery cycle involves:

1. Courier rides an electric scooter to reach the restaurant after receiving the order
2. Upon arrival, walks into the restaurant to pick up the food
3. May remain stationary while waiting if food is not ready
4. Rides the scooter and finally walks to the customer's location

Accurately identifying time points (such as arrival at restaurant) is crucial for the online platform to optimize its decisions. Activity status provides strong clues for detecting these time points and tracking the courier's progress.

We collaborated with **Ele.me**, the second-largest on-demand food delivery platform in China that operates in over 300 cities and employs around half a million couriers. Building upon the original LIMU-BERT design, we leverage IMU sensors embedded in smartphones to recognize the activity and movement status of couriers.

### Three Deployment Phases

| Phase | Period | Scope |
|-------|--------|-------|
| Phase I | Jan 2022 – Jun 2022 | Feasibility study in Yangzhou City |
| Phase II | Jul 2022 – Dec 2022 | Pretraining on billions of unlabeled samples (60K couriers, 1.1K phone models) |
| Phase III | Jun 2023 – present | Nationwide deployment to 500K couriers across 367 cities |

Currently, models execute approximately **7.5 billion predictions daily**.

### Business Impact

Comprehensive end-to-end A/B testing using over 1 million orders showed:
- Mean absolute error (MAE) of Estimated Time of Stop (ETS) reduced by **1.8 seconds per order**
- New pricing strategy saves an average of **0.06 yuan per delivery**
- Estimated annual savings of approximately **0.44 billion RMB**

### Contributions

- Showcase the practical value of leveraging human activity recognition technology in real-world commercial applications
- Share experiences and lessons learned in training, evaluating, and deploying deep learning-based HAR models for large-scale scenarios
- Open-source LIMU-BERT pretrained with approximately 1.43 million hours of sensor data from 60K subjects and 1.1K phone models (https://github.com/WANDS-HKUST/LIMU-BERT_Experience)

---

## 2. Activity Recognition for On-demand Food Delivery Service

### 2.1 On-demand Food Delivery

On-demand delivery is a rapidly growing industry where couriers transport meals and groceries directly from merchants to customers. Platforms require intelligent order dispatch strategies that can optimally match couriers with food orders in real-time.

The food readiness or pickup times reported by couriers are often subject to objective bias. The courier's activity context provides a natural segmentation of their trajectory, aiding in pinpointing key time points within specific time ranges.

### 2.2 Challenges

The primary challenge lies in the absence of a comprehensive and large-scale labeled sensor dataset. In the ele.me application scenario:
- Over 500K couriers across 367 cities in China
- More than 1,000 different phone models
- Significant data heterogeneity due to variations in usage patterns, devices, and environmental factors

While numerous public IMU datasets exist, none address the level of diversity required for large-scale applications. Most were collected under controlled conditions, with smart devices fixed at specific positions. In contrast, real-world scenarios reveal that smartphone placement frequently varies, and users interact with their devices during use.

#### Performance Degradation Across Datasets

| Training Dataset | Test Dataset | Accuracy | F1-score |
|------------------|--------------|----------|----------|
| Shoaib | Shoaib | 89.5% | 89.2% |
| Shoaib | Yangzhou | 28.2% | 20.8% |

This highlights that data discrepancies can severely hinder the model's ability to generalize to real-world scenarios.

### 2.3 Activity Recognition Design

We adopt **LIMU-BERT**, a state-of-the-art IMU foundation model, which effectively harnesses the potential of large-scale unlabeled data. The two distinct learning phases:

1. **Pre-training Phase:** LIMU-BERT is pre-trained using extensive unlabeled data. We randomly mask a subset of the IMU sequence data (e.g., 20 out of 120 samples) and jointly train LIMU-BERT along with a decoder to reconstruct the partially masked inputs.

2. **Fine-tuning Phase:** The pre-trained LIMU-BERT is reused and connected to a classifier model. Together, they are fine-tuned using a limited amount of labeled data for activity recognition tasks.

The original model consists of only **71K parameters**, making it lightweight and capable of supporting real-time activity recognition on smartphones.

---

## 3. Deployment and Experience

### Three-Phase Deployment Overview

| Phase | Phase I (Small-scale eval) | Phase II (Large-scale eval) | Phase III (Online deployment) |
|-------|---------------------------|----------------------------|-------------------------------|
| Timeline | 2022/01-2022/06 | 2022/07-2022/12 | 2023/01-present |
| Area | sub-city | 367 cities | 367 cities |
| Labeled Training | 10 couriers, 8 phone models, 822K samples | 10 couriers, 8 phone models, 902K samples | - |
| Unlabeled Training | - | 60K couriers, 1.1K phone models, 858M samples | - |
| Evaluation | 10 couriers, 8 phone models, 97k samples | 500K couriers, 1.9K phone models, 290M samples | - |
| Deployment | - | 5 couriers, 4 phone models, 3K samples | 500K couriers, 1.9K phone models, 7.5B samples/day |
| Performance | 89.2% accuracy, 88.1% F1-score (3-activity classification) | 90.1% precision, 94.5% recall (riding/non-riding) | - |

### 3.1 Phase I: Small-scale Evaluation

#### Design Customization

Several adjustments to better suit the scenario:

1. **Sampling rate reduced** from 20 Hz to 10 Hz (lower data collection costs, reduced sequence length)
2. **Window size reduced** from 120 (12 seconds at 10 Hz) to 60 (6 seconds at 10 Hz)
3. **Joint fine-tuning** — all parameters in LIMU-BERT are fine-tuned with the classifier
4. Model definition: R_num = 4, A_dim = 4, H_dim = 36, F_dim = 72
5. GRU classifier with three stacked layers (hidden sizes 20, 20, 10)
6. Training: 700 epochs, learning rate 0.001, batch size 128

#### Impact of Sampling Rate

| Dataset | Sampling Rate | Accuracy | F1-score |
|---------|---------------|----------|----------|
| Shoaib | 20Hz | 90.0% | 90.1% |
| Shoaib | 10Hz | 89.5% | 89.2% |

#### Yangzhou Dataset Collection

A video-assisted approach was used:
- Couriers equipped with wearable cameras mounted on chests
- ~420 hours of video footage analyzed
- Third-party company engaged for annotation
- 10 delivery riders with 8 unique smartphone models for 7 days
- ~13,000 samples per rider per day
- Total samples: 277,140 (still), 168,007 (walking), 456,629 (riding) = 901,776 samples
- Three-class classification: still, walking, and riding

#### Performance Comparison on Yangzhou Dataset

| Method | Accuracy | F1-score |
|--------|----------|----------|
| FFT+LR | 81.4% | 80.1% |
| DCNN | 80.6% | 79.7% |
| **LIMU-BERT** | **89.2%** | **88.1%** |

Pretraining LIMU-BERT on Shoaib dataset and fine-tuning on Yangzhou dataset degraded performance significantly (75.9% accuracy, 75.6% F1-score), emphasizing the importance of collecting a large-scale unlabeled dataset tailored to the application scenario.

### 3.2 Phase II: Large-scale Evaluation

#### Unlabeled Dataset Collection

- 847,684,084 samples
- 1.1K smartphone models
- 60K couriers across China

#### Model Re-training

- Pre-training on four Tesla-V100-32G GPUs
- Batch size: 128
- Epochs: 8,000
- Loss: MSE between original and reconstructed data
- Optimizer: Adam, learning rate 0.001
- Total training time: ~780 GPU hours

#### Performance with Pretraining vs Without

| Labeling rate | 0.10% | | 1% | | 10% | | 20% | | 90% | |
|---------------|-------|-----|-----|-----|------|------|------|------|------|------|
| Metric | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 |
| LIMU-BERT w/ pretraining | 70.10% | 66.50% | 81.20% | 80.20% | 87.60% | 86.20% | 88.90% | 87.8% | 90.50% | 89.60% |
| LIMU-BERT w/o pretraining | 64.50% | 60.90% | 76.90% | 75.80% | 84.40% | 83.10% | 86.6% | 85.40% | 89.20% | 88.10% |

The pretraining-enabled model consistently delivers better results across all cases.

#### Small-scale Deployment

- 5 couriers with 4 different phone models
- ~3,000 samples for three-class classification
- Accuracy: 88.82%, F1 score: 83.07%

#### Large-scale Evaluation (Riding/Non-riding)

A rule-based approach derived labels from the courier's environment:
- **Riding:** Outdoors with GPS speed > 4 m/s
- **Non-riding:** Indoors

Results on 290M samples:
- Precision (riding): **90.1%**
- Recall (riding): **94.5%**

### 3.3 Phase III: Online Deployment

#### SDK Modules

1. **Data collection module** — listening service for raw sensor data, including barometers and satellite signals
2. **Edge computing module** — manages on/off functionality and performs activity prediction using locally collected data
3. **Control module** — dynamic configuration of settings, data compression, uploads, and local caching

#### Package Size Optimization

- Initial PyTorch Lite compilation: **+13 MB increase** (unacceptable)
- Manual Android PyTorch compilation with dynamic loading strategy: **+100 KB only**
- Large SO libraries fetched from server at runtime when needed

#### SDK Power Consumption

| Active SDK Modules | Voltage (V) | Current (mA) | Power (mW) |
|--------------------|-------------|--------------|------------|
| All | 4.2 | 432.05 | 1814.61 |
| All excluding edge computing | 4.2 | 429.05 | 1802.01 |
| None | 4.2 | 383.08 | 1608.94 |

Total power increase: ~13%, with ~10% attributed to continuous satellite signal search and <1% to SDK computational processes.

#### Launch Details

- Deployment began December 2023
- Currently supports ~500,000 daily active couriers
- 367 cities covered
- ~20 million food delivery orders processed daily
- LIMU-BERT operates on ~1,900 distinct phone models
- Predictions every two seconds
- **~7.5 billion predictions per day**

---

## 4. Downstream Applications and Business Benefits

### 4.1 Trajectory Segmentation and Navigation

#### Algorithm 1: Trajectory Segmentation with Activity Recognition Integration

```
Step 1: Smooth Activity States
1: acts ← []
2: for each point in trajectory do
3:     acts.append(point.getActState())
4: end for
5: smoothActs ← smoothActs(acts, window_size)

Step 2: Identify Activity Clusters
6: actClusters ← findActClusters(smoothActs)

Step 3: Refine GPS Data
7: refinedGPS ← refineGPS(actClusters, trajectory)

Step 4: Segment GPS Data
8: finalSegs ← segmentGPS(refinedGPS, actClusters)
9: return finalSegs
```

Evaluation results:
- Manual labeling on 1,000 delivery orders: **95.2% classification accuracy**
- Transition point precision over 20 million orders: **88.7%**

**Navigation:** Leveraging crowdsourced trajectories and accurate segmentation results provides more precise recommendations for food pick-up and drop-off points.

### 4.2 Elevation Change Detection

Only 8% of Android phones are equipped with barometer sensors, but the majority of iOS devices have them. We propose a **device collaboration training approach**:

1. **First learning phase:** Use barometer data from smartphones with barometers to generate two-class labels (vertical/non-vertical movements)
   - Vertical movement: air pressure change ≥ 0.25 hPa, change speed > 0.016 hPa/sec
2. **Second learning phase:** Deploy trained model on smartphones without barometers to detect vertical movement using IMU data alone

Results:
- Fine-tuned with ~700K labeled samples
- Evaluated on ~97K samples
- **Accuracy: 82.5%**, **F1 score: 82.4%**

### 4.3 Estimated Time of Stop (ETS)

Dataset constructed from major urban areas (Shenzhen, Xi'an, Guangzhou):
- Training: 2.38 million orders
- Validation: 1.03 million orders
- Test: 780 thousand orders
- Period: mid-May to early June 2024

#### Improved Predictions in ETS

| Category | MAE Reduction (s) | Under-estimation Rate Reduction (%) | Over-estimation Rate Reduction (%) |
|----------|-------------------|--------------------------------------|-------------------------------------|
| Overall | 1.8 | 4.98 | 4.57 |
| Walk-only AOI | 3.6 | 3.92 | 9.34 |
| Difficult AOI | 2.4 | 16.08 | 5.43 |

A "difficult" AOI refers to an area where deliveries are challenging due to long walking distances or prolonged elevator wait times.

### 4.4 Difficulty Analysis and Pricing Strategy

A/B test conducted in Shanghai (July 8-14, 2024):
- Experimental group: 510,000 orders
- Control group: 2.02 million orders

Results:
- Average basic delivery fee per order reduced by **0.06 yuan**
- Order acceptance rate maintained within first five minutes
- With ~20 million orders processed daily, this translates to approximately **0.44 billion RMB annual savings**

---

## 5. Lessons Learned

### 5.1 Scaling Law

A scaling analysis using up to 1.43 million hours of IMU data from over 50,000 couriers shows the MSE loss steadily decreases as training samples and model parameters increase. However, a saturation effect was observed.

Final configuration:
- Dataset: 852 million samples
- Model parameters: 137 thousand

The findings confirm that the scaling law also applies to IMU foundation models.

### 5.2 Evaluating Models at Large Scale

Vision-assisted approaches provide highly accurate labels but are time-consuming and expensive:
- 420 hours of video footage annotation took over two weeks
- Cost more than 100,000 RMB

The rule-based approach using IODetector and GPS speed proved cost-effective and scalable, despite potentially introducing noisy labels.

### 5.3 Handling Diverse Devices

#### Sensor Availability Coverage

- Initial coverage: 89% of couriers
- ~57,000 devices excluded due to missing gyroscope
- Solution: Developed accelerometer-only LIMU-BERT
  - Performance: 87.4% accuracy, 86.3% F1 score (three-class)
- Final coverage: ~99% of couriers

#### Synchronization

10 Hz sampling frequency considered:
- Model performance and transmission overhead
- Device-specific factors (sensor data updates not always simultaneous)
- A synchronization component ensures all sensor data are properly aligned

#### SDK Success Rates

- Most models: success rates above 99%
- Less common brands (TECNO, Cancro): below 90%, likely due to lower system compatibility

### 5.4 Sensor and Device Collaboration

- Crowdsourced labeling relies on GPS speed and IODetector (light + magnetometer sensors)
- Elevation change detection model uses barometer readings
- iOS devices' barometers used to train models for Android smartphones
- Some behaviors cannot be fully distinguished using IMU data alone:
  - Riding on flat terrain at constant speed resembles standing still
  - Subtle walking movements may resemble "still" state during typing
- Hybrid approaches integrating additional sensor data (e.g., GPS) can help

### 5.5 Others

- Dynamic loading strategy critical for reducing app size
- Data augmentation (random scaling and rotation) provided only marginal improvements:
  - Accuracy slightly decreased by 1.24%
  - F1 score increased by 1.74%
- Augmentation effectiveness diminishes when training data is extensive

---

## 6. Discussion

**Privacy concerns.** The platform relies on privacy-sensitive data such as location for efficient order assignment. Couriers agree to share this data during their working hours as part of their contractual agreement.

**Potential generalization.** The working patterns of couriers are distinct from those in other professions. Generalization to other domains may require fine-tuning, but LIMU-BERT offers a strong foundation requiring only modest amounts of labeled data and training time.

**Future work.** Exploring more applications of activity recognition, such as detecting elevator/lift usage in buildings to support more accurate time estimation and difficulty analysis.

---

## 7. Related Work

**Real-world mobile system experience.** Previous work has reported experiences with large-scale mobile system deployments including wrist-worn computing, indoor localization, and indoor-outdoor detection. This is the **first nationwide deployment of deep learning-based models on mobile devices for human activity recognition**.

**Activity recognition.** Wearable-based solutions are more ubiquitous and cost-effective than image-based or wireless approaches. Recent self-supervised learning techniques enable foundation models that leverage inexpensive and easily accessible data.

**On-demand food delivery.** Most works focus on extracting location contexts of couriers (indoor status detection, merchant-level localization) or time inference for order servicing. This is the first work to deploy nationwide activity recognition for on-demand food delivery.

---

## 8. Conclusion

This paper presents our experience in adopting human activity recognition (HAR) technology to support the real-world business of on-demand food delivery. We share our deployment experience, the lessons learned, and the practical considerations for adopting research innovations to large-scale commercial applications, paving the way for future advancements in mobile computing technologies.

---

## Acknowledgments

We thank all reviewers for their insightful comments. This work is supported by the Global STEM Professorship Scheme of Hong Kong and the HKUST start up grant. Mo Li is the corresponding author.

---

## References

1. Fengniao delivery. https://fengniao.ele.me/
2. Özgü Alay et al. 2017. Experience: An open platform for experimentation with commercial mobile broadband networks. In MobiCom. 70–78.
3. David G Andrews. 2010. An introduction to atmospheric physics. Cambridge University Press.
4. George Boateng et al. 2019. Experience: Design, development and evaluation of a wearable device for mHealth applications. In MobiCom. 1–14.
5. Youngjae Chang, Akhil Mathur, Anton Isopoussu, Junehwa Song, and Fahim Kawsar. 2020. A systematic study of unsupervised domain adaptation for robust human-activity recognition. IMWUT 4, 1 (2020), 1–30.
6. Zhigang Dai, Wenjun Lyu, Yi Ding, Yiwei Song, and Yunhuai Liu. 2023. OPTI: Order Preparation Time Inference for On-demand Delivery. ACM TOSN 19, 4 (2023), 1–18.
7. Yi Ding et al. 2022. P2-loc: A person-2-person indoor localization system in on-demand delivery. IMWUT 6, 1 (2022), 1–24.
8. Yi Ding et al. 2021. Nationwide deployment and operation of a virtual arrival detection system in the wild. In SIGCOMM. 705–717.
9. Baoshen Guo et al. 2022. Wepos: Weak-supervised indoor positioning with unlabeled wifi for on-demand delivery. IMWUT 6, 2 (2022), 1–25.
10. Harish Haresamudram, Irfan Essa, and Thomas Plötz. 2022. Assessing the state of self-supervised human activity recognition using wearables. IMWUT 6, 3 (2022), 1–47.
11. Tom Hoddes et al. 2025. Scaling laws in wearable human activity recognition. arXiv preprint arXiv:2502.03364.
12. Zhiqing Hong et al. 2024. CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining. IMWUT 8, 2 (2024), 1–26.
13. Yuming Hu et al. 2022. Experience: Practical indoor localization for malls. In MobiCom. 82–93.
14. Ashish Jaiswal et al. 2020. A survey on contrastive self-supervised learning. Technologies 9, 1 (2020), 2.
15. Wenchao Jiang and Zhaozheng Yin. 2015. Human activity recognition using wearable sensors by deep convolutional neural networks. In ACM MM. 1307–1310.
16. Yonghang Jiang, Zhenjiang Li, and Jianping Wang. 2018. Ptrack: Enhancing the applicability of pedestrian tracking with wearables. IEEE TMC 18, 2 (2018), 431–443.
17. Jared Kaplan et al. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.
18. Denizhan Kara et al. 2024. PhyMask: An Adaptive Masking Paradigm for Efficient Self-Supervised Learning in IoT. In SenSys. 97–111.
19. Hyung-Sin Kim, JeongGil Ko, and Saewoong Bahk. 2017. Smarter markets for smarter life: Applications, challenges, and deployment experiences. IEEE Communications Magazine 55, 5 (2017), 34–41.
20. Charlene Li, Miranda Mirosa, and Phil Bremer. 2020. Review of online food delivery platforms and their impacts on sustainability. Sustainability 12, 14 (2020), 5528.
21. Yuanjie Li et al. 2021. Experience: a five-year retrospective of MobileInsight. In MobiCom. 28–41.
22. Shengzhong Liu et al. 2020. GIobalFusion: A Global Attentional Deep Learning Framework for Multisensor Information Fusion. IMWUT 4, 1 (2020), 1–27.
23. Wei Liu et al. 2022. Para-pred: Addressing heterogeneity for city-wide indoor status estimation in on-demand delivery. In KDD. 3407–3417.
24. Xiao Liu et al. 2021. Self-supervised learning: Generative or contrastive. IEEE TKDE 35, 1 (2021), 857–876.
25. Yang Liu, Zhenjiang Li, Zhidan Liu, and Kaishun Wu. 2019. Real-time arm skeleton tracking and gesture inference tolerant to missing wearable sensors. In MobiSys. 287–299.
26. Aleksej Logacjov. 2024. Self-supervised learning for accelerometer-based human activity recognition: A survey. IMWUT 8, 4 (2024), 1–42.
27. Mohammad Malekzadeh, Richard G Clegg, Andrea Cavallaro, and Hamed Haddadi. 2019. Mobile sensor data anonymization. In IoTDI. 49–58.
28. Girish Narayanswamy et al. 2024. Scaling wearable foundation models. arXiv preprint arXiv:2410.13638.
29. Jiazhi Ni et al. 2022. Experience: Pushing indoor localization from laboratory to the wild. In MobiCom. 147–157.
30. Xiaomin Ouyang et al. 2021. ClusterFL: a similarity-aware federated learning system for human activity recognition. In MobiSys. 54–66.
31. Adam Paszke et al. 2019. Pytorch: An imperative style, high-performance deep learning library. NeurIPS 32 (2019).
32. Hangwei Qian, Tian Tian, and Chunyan Miao. 2022. What makes good contrastive learning on small-scale wearable-based tasks? In KDD. 3761–3771.
33. Zhen Qin et al. 2019. Learning-aided user identification using smartphone sensors for smart homes. IEEE IoT Journal 6, 5 (2019), 7760–7772.
34. Jorge-L Reyes-Ortiz et al. 2016. Transition-aware human activity recognition using smartphones. Neurocomputing 171 (2016), 754–767.
35. Aaqib Saeed, Tanir Ozcelebi, and Johan Lukkien. 2019. Multi-task self-supervised learning for human activity detection. IMWUT 3, 2 (2019), 1–30.
36. Philip Sedgwick and Nan Greenwood. 2015. Understanding the Hawthorne effect. BMJ 351 (2015).
37. Spencer Sevilla, Matthew Johnson, Pat Kosakanchit, Jenny Liang, and Kurtis Heimerl. 2019. Experiences: Design, implementation, and deployment of CoLTE, a community LTE solution. In MobiCom. 1–16.
38. Zhiyao Sheng, Huatao Xu, Qian Zhang, and Dong Wang. 2022. Facilitating radar-based gesture recognition with self-supervised learning. In SECON. IEEE, 154–162.
39. Muhammad Shoaib et al. 2014. Fusion of smartphone motion sensors for physical activity recognition. Sensors 14, 6 (2014), 10146–10176.
40. Allan Stisen et al. 2015. Smart devices are different: Assessing and mitigating mobile sensing heterogeneities for activity recognition. In SenSys. 127–140.
41. Scott Sun, Dennis Melamed, and Kris Kitani. 2021. IDOL: Inertial Deep Orientation-Estimation and Localization. In AAAI 35. 6128–6137.
42. Timo Sztyler and Heiner Stuckenschmidt. 2016. On-body localization of wearable devices: An investigation of position-aware activity recognition. In PerCom. IEEE, 1–9.
43. Chi Ian Tang et al. 2021. SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data. IMWUT 5, 1 (2021), 1–30.
44. Jinqiang Wang et al. 2022. Sensor Data Augmentation by Resampling in Contrastive Learning for Human Activity Recognition. IEEE Sensors Journal 22, 23 (2022), 22994–23008.
45. Huatao Xu, Pengfei Zhou, Rui Tan, and Mo Li. 2023. Practically Adopting Human Activity Recognition. In MobiCom. 1–15.
46. Huatao Xu, Pengfei Zhou, Rui Tan, Mo Li, and Guobin Shen. 2021. LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications. In SenSys. 220–233.
47. Xiangyu Xu et al. 2020. TouchPass: towards behavior-irrelevant on-touch user authentication on smartphones leveraging vibrations. In MobiCom. 1–13.
48. Jianbo Yang et al. 2015. Deep convolutional neural networks on multichannel time series for human activity recognition. In IJCAI 15. 3995–4001.
49. Shuochao Yao et al. 2017. Deepsense: A unified deep learning framework for time-series mobile sensing data processing. In WWW. 351–360.
50. Han Zhou et al. 2019. LimbMotion: Decimeter-level Limb Tracking for Wearable-based Human-Computer Interaction. IMWUT 3, 4 (2019), 1–24.
51. Pengfei Zhou et al. 2022. Experience: Adopting indoor outdoor detection in on-demand food delivery business. In MobiCom. 94–105.
52. Pengfei Zhou, Yuanqing Zheng, Zhenjiang Li, Mo Li, and Guobin Shen. 2012. Iodetector: A generic service for indoor outdoor detection. In SenSys. 113–126.
53. Lin Zhu et al. 2020. Order fulfillment cycle time estimation for on-demand food delivery. In KDD. 2571–2580.
