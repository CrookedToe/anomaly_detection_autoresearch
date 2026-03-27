---
id: lakey-deep-architectures-spacecraft-2024
type: preprint
title: "A Comparison of Deep Learning Architectures for Spacecraft Anomaly Detection"
authors: "Lakey, Daniel; Schlippe, Tim"
year: 2024
venue: "arXiv"
doi: ""
url: "https://arxiv.org/abs/2403.12864"
arxiv: "2403.12864"
tags:
  - spacecraft
  - anomaly-detection
  - deep-learning
  - architecture-comparison
added: 2026-03-26
source: manual
---

# A Comparison of Deep Learning Architectures for Spacecraft Anomaly Detection

## Abstract

Comparative study of **thirteen** deep learning architectures for spacecraft telemetry anomaly detection on the public **NASA SMAP/MSL** benchmark used with **Telemanom** [4]. Models span CNN, RNN, Transformer, MLP, wavelet, and hybrid families (via `tsai`). Channels are clustered with **K-means** on statistical moments; best single model **XceptionTimePlus** reaches ~**69.9%** F1 (anomaly); picking the best architecture **per cluster** yields ~**84.7%** average F1, slightly above the Telemanom baseline (~83.6%) under their experimental constraints.

## Summary

- **Comparative** study of deep learning for **spacecraft telemetry** anomaly detection: replaces Telemanom’s default LSTM with **13** architectures from **`tsai`** (CNN / RNN / Transformer / hybrid / mWDN / gMLP, etc.) on the **SMAP/MSL** dataset (**82** channels, one model per channel).
- **K-means** clustering of channels by moments of the target parameter motivates **per-cluster** model choice; best **global** F1 anomaly ~**69.9%** (**XceptionTimePlus**); **per-cluster** best mix ~**84.7%** vs Telemanom baseline **83.6%** (their Table 1), without heavy hyperparameter search.
- Relevant when comparing **TCN/CNN-style** encoders to other families for telemetry-style multivariate forecasting + thresholding pipelines (as in this repo’s TCN + memory gating).

## Full text (optional)

*Canonical PDF/HTML: [arXiv:2403.12864](https://arxiv.org/abs/2403.12864). The body below is pasted Markdown for local reading; prefer the arXiv version for citation.*

## 1. Introduction

The field of space exploration has had significant advancements in recent decades, characterised by the increasing sophistication of spacecraft and the expanding complexity of missions. As mankind expands its presence in outer space, the importance of precise and dependable data from spacecraft systems has become of utmost significance. Time series data, which refers to a sequential arrangement of data points organised in chronological order, holds major significance in the domain of spacecraft telemetry. Spacecraft systems are reflected by telemetry data, which provides information on their state, health, and performance. This data allows for the analysis of both regular and potentially abnormal operations [1].

Anomalies observed in spacecraft telemetry data are unanticipated occurrences that pose potential risks, as they depart significantly from the predicted operational patterns of the system. The quick detection and identification of these abnormalities is of paramount significance in order to avert catastrophic failures, limit risks, and guarantee the durability of space missions. According to [2], the prompt identification and effective detection of these anomalies by operational engineers play a crucial role in enhancing efficiency, minimising expenses, and enhancing safety. As the complexity of spacecraft continues to advance, there is a corresponding growth in the variety of telemetry parameters associated with them. The utilisation of conventional, manual, or simple “out-of-limits” techniques are becoming ever more difficult for the purpose of identifying anomalies [3].

In recent years, there has been considerable focus on the advancement of anomaly detection techniques for satellite telemetry data. Numerous advanced algorithms and strategies have been proposed by prominent organisations such as NASA [4], ESA [3], and CNES [5] to tackle this task. Every approach possesses its own set of advantages and disadvantages. There is a clear trend towards deep learning approaches over statistical methods due to their ability to synthesise the complex multivariate temporally-connected data inherent to spacecraft telemetry [6]. The objective of this paper is to investigate and assess different methodologies for anomaly identification in order to determine the most optimal and efficient approach for analysing spacecraft telemetry.

Our work pioneers several notable contributions to the domain of spacecraft anomaly detection, presenting advancements that enhance the understanding of deep learning in this field. Firstly, it unfolds a comprehensive side-by-side comparison of multiple deep learning model architectures, shedding light on their effectiveness in detecting anomalies in spacecraft telemetry. This comparison is distinctively valuable as it incorporates models that, to our knowledge, have not been previously applied to spacecraft anomalies, thereby opening new avenues for exploration and implementation. Secondly, we introduce an innovative unsupervised mechanism to cluster spacecraft telemetry into like-types, using statistical methods, which allows for a more granular and nuanced understanding of telemetry data. Thirdly, our study unveils insights into the comparative performance of different deep learning models across the identified clusters, providing insights for selecting the most suitable model based on the specific type of telemetry data. These diverse contributions collectively elevate the current state of research in spacecraft anomaly detection, offering robust and refined tools and methodologies for practical applications and future explorations.

### Nomenclature

- **Telemetry Channel:** A specific pathway or conduit used for transmitting telemetry data [7], for example from a specific sensor on the spacecraft. A telemetry channel consists of one or more parameters.  
- **Parameter:** A measurement within a telemetry channel. This may be an analogue reading such as temperature or current, a discrete numerical value, or a binary status. A telemetry time series is made up of many samples of a number of parameters representing a number of telemetry channels.  
- **Dataset:** A collection of data points or individual pieces of information, organised usually in tabular form, where rows represent individual records and columns represent attributes or variables of the data. The dataset used in our study is from [4] and contains data for 82 telemetry channels.  
- **Cluster:** A grouping of data points or items in a dataset that share similar characteristics or properties, typically identified through various methods of cluster analysis, allowing for the study of relationships and patterns within the data [8].  

---

## 2. Related Work

This section will describe related work investigated as part of our study, focusing especially on popular deep learning architectures, many of which have not been applied to the problem of anomaly detection in spacecraft telemetry data.

### Data for Spacecraft Anomaly Detection

Modern spacecraft have many thousands of telemetry channels [9], and this “huge” [10] amount of data is more than can be monitored by human operators. Within these channels, actual instances of anomalies are rare. By design a spacecraft is a robust machine, fault tolerant and extensively tested to ensure that anomalies do not occur [11]. For example, a study of seven different spacecraft over more than a decade yielded fewer than 200 critical anomalies [12].

Spacecraft Anomaly Detection is a particularly challenging field due to the sparsity of publicly available datasets for training. Indeed, of all the studies listed in our work, only [4] make the data available, and even then with implementation-specific details hidden through scaling and normalisation. This has led to their dataset becoming a benchmark for further studies, such as [9], and consequently we used it in our experiments.

The dataset provided by [4] comprises 82 telemetry channels taken from the Soil Moisture Active Passive (SMAP) [13] spacecraft and “Curiosity” Mars Science Laboratory (MSL) [14] spacecraft. In Section 3, we will describe this dataset in the context of our experimental setup.

### Approaches for Spacecraft Anomaly Detection

A typical approach, for example followed by [3], [4], [5], and [9], is the use of deep learning models to perform regression-based forecasting on a time series and identify anomalies by comparing predictions to real values received from the spacecraft. The central concept is “to reconstruct the telemetry sequence based on training data, and anomalies are identified if the reconstruction errors exceed a given threshold.” [15], as illustrated in Figure 1. “The idea is to use past telemetry describing normal spacecraft behaviour in order to learn a reference model to which can be compared most recent data in order to detect potential anomalies.” [5]. Multivariate models are used to capture spatial and temporal linkages between separate telemetry channels [16].

Whilst effective, it relies on the selection of some threshold value beyond which the construction error is considered anomalous. [4] propose “Telemanom”, an “unsupervised and nonparametric anomaly thresholding approach” where the anomaly detector dynamically learns the error value corresponding to the anomaly for a particular time series. They report excellent F1 scores for the anomaly detection, as synthesised in Table 1, which we use as our baseline.

### Table 1. Telemanom F1 Scores

| Dataset | F1 Score |
|---|---:|
| SMAP | 85.5% |
| MSL | 79.3% |
| Total | 83.6% |

There exist many types of architecture for deep learning, many of which have been tuned specifically to time series prediction, for example [17]. We selected six state-of-the-art families of architecture for further investigation. Additionally, we investigated two hybrid architectures comprised of a combination of two or more model types as suggested by [18].

### Chosen Deep Learning Architectures for our Study

The following sections briefly review the chosen deep learning architectures used in our study. In particular, it is noted whether previous work has tried these in the domain of spacecraft anomaly detection and which are novel to this task.

#### Multilevel Wavelet Decomposition

[19] introduced their Multilevel Wavelet Decomposition Network (mWDN) for anomaly detection. mWDN leverages the benefits of wavelet transformation in conjunction with a deep learning model to analyse time series data, with a specific emphasis on interpretability. Wavelet transformation is a powerful mathematical tool often used for analysing different frequency components in time series data, which makes it highly suitable for anomaly detection in varied applications such as high-frequency signals [20] and power converters [21]. To the best of our knowledge, an mWDN has yet to be applied to spacecraft telemetry anomaly detection.

#### Multi-Layer Perceptron (MLP)

The gMLP [22], or gated Multi-Layer Perceptron, is a type of artificial neural network model designed to have performance competitive to Transformer models but with a more straightforward architecture. It relies more on feedforward layers and less on attention mechanisms. A gMLP utilises Spatial Gating Units (SGU), a central component that enables information exchange between different positions in the sequence, allowing the model to capture dependencies between different parts of the input. MLPs have been used for anomaly detection in fields as varied as water treatment [23] to rogue trading [24], as well as spacecraft anomaly detection [25].

#### Transformer

Transformers, originally proposed in [26], are a type of neural network architecture that have become the foundation for most state-of-the-art models in natural language processing, and they are increasingly being used in various domains like time series analysis and image processing. Transformers use a mechanism called self-attention that allows each element in the input sequence to consider other elements in the sequence when producing its output, weighting each one differently depending on the learnt relationships. Transformers are the subject of much active research into anomaly detection, such as [27], [28], and [29]. The implementation of the Transformer architecture investigated in our study is TimeSeriesTransformer (TST) [30], which tunes the architecture specifically for multivariate time series data, of which spacecraft telemetry is an extreme example owing to the potentially very large number of parameters to consider [31].

#### Convolutional Neural Network (CNN)

Convolutional Neural Networks (CNNs) are a class of deep learning models primarily developed for analysing visual imagery, renowned for their ability to learn hierarchical features from input data [32]. In the context of spacecraft anomaly detection, CNNs can be utilised to process multivariate time series data generated by spacecraft sensors [33], enabling the identification of anomalous patterns or events indicative of potential faults, malfunctions, or other abnormalities in spacecraft systems [34]. By learning both spatial and temporal features in the data, CNNs can aid in early and accurate detection of anomalies [35].

As a popular architecture in deep learning, there are many implementations of interest. Four are selected here, including the “classics” ResNet [36], [37] and Fully Convolutional Network (FCN) [37], in addition to some implementations specifically tailored to time series: XceptionTime [38] and InceptionTime [39]. For InceptionTime, two implementations from `tsai` were chosen: `MultiInceptionTimePlus` and `InceptionTimeXLPlus`. The former is an ensemble method with multiple internal models, whereas the latter contains a large number of parameters.

#### Recurrent Neural Network (RNN)

Recurrent Neural Networks (RNNs) are a category of neural networks specialised for processing sequential data, enabling the modeling of temporal features within the sequences. Our study includes two RNN variants. Long Short-Term Memory (LSTM) [40] units are a variant of RNNs designed to mitigate the vanishing and exploding gradient problems inherent in basic RNNs, allowing them to learn long-range behaviours within the data. The model used in our baseline study [4] is LSTM-based. Gated Recurrent Units (GRU) [41] are another variant of RNNs, similar to LSTMs but with a simpler structure, designed to capture dependencies for sequences of varied lengths. GRUs have been proven to perform comparably to LSTMs on certain tasks [42] but with reduced computational requirements, offering an efficient alternative for sequence modeling. This may make them of particular use in spacecraft anomaly detection [43], where the cost of training the more complex models may be prohibitive.

#### Hybrid Models

To leverage the advantages of different deep learning architectures, many previous studies have considered hybrid models [44], [18]. LSTM/Transformer models are quite popular, for example [45] and [46], and in our study we include two such models in the suite of tested architectures, `TransformerLSTM` [47], [48] and `LSTMAttention` [30], [26], [48]. Other studies [49] have considered a hybrid FCN/Transformer model, combining the spatial learning abilities of CNNs with the sequence learning of Transformers, therefore we include `LSTM_FCN` [48] in our test set. Other studies such as [50] go further still in combining CNN, RNN, and Transformer architectures in one model.

We are unaware of the use of hybrid deep learning models for spacecraft anomaly detection, although there have been studies in the area of hybrid machine learning such as [15] and [51].

---

## 3. Experimental Setup

This section describes the details of the experimental setup used for the comparison of deep learning approaches.

### Implementation

We employed the Telemanom [4] anomaly detection framework for conducting experiments on spacecraft anomaly detection. The original implementation of Telemanom uses a TensorFlow-based LSTM model as a default, designed to recognize anomalous patterns in time series data relevant to spacecraft telemetry. We replace this LSTM with alternative architectures as described in Section 2.

We retain the dynamic thresholding and anomaly detection algorithms of Telemanom whilst replacing the time-series forecasting models with a variety of new architectures. Thus, we can clearly demonstrate the differences due to the architecture alone. To this end, the default LSTM was replaced with various models provided by the `tsai` library [48], a PyTorch- and fastai-based collection of time-series deep learning architectures [52]. The `tsai` library implements a wide selection of state-of-the-art models optimised for time-series data. In order to keep the model code generic and easily fit to a variety of different model architectures, the `tsai` “Plus” implementations of the above architectures were selected due to their common interfaces. Detailed documentation regarding the particulars of implementation can be found in [48].

Following the approach taken in [4], one model is trained per telemetry channel. Our study compares thirteen different architectures, leading to `82 × 13 = 1,066` trained models overall.

The models were trained utilizing the “fit one cycle” method [53], a technique noted for its efficacy in training deep learning models efficiently and reliably. The experiment endeavoured to keep the setup fair and comparable; thus, hyperparameter tuning was predominantly confined to ensuring that the RNN-based architectures possessed at least equivalent depth to the default LSTM implemented in Telemanom. Apart from this modification, we retained the default hyperparameters provided by the `tsai` and `fastai` models to maintain the integrity of the comparative analysis, on the basis that the defaults are anyway sensible [54]. Furthermore, due to the large number of trained models, hyperparameter tuning was infeasible in any case.

Early experience during model training showed that model performance was very sensitive to the learning rate. In order to negate these effects, we applied the learning rate reduction scheme `ReduceLROnPlateau`, provided by the `fastai` framework [52], to each model. The callback reduces the learning rate on each epoch if the training loss metrics are unchanging between consecutive epochs. This has given good results in studies such as [55] but at the cost of longer training times.

The computational environment for the experiments was provisioned on a virtual machine, equipped with 8 CPU cores (Intel Xeon Platinum 8260 CPU @ 2.40GHz) and 16GB of RAM. No hardware acceleration or GPUs were available.

### Data

Due to commercial, legal, and security considerations, there are very few well-labelled spacecraft anomaly datasets available to the public. The “SMAP/MSL” dataset provided by [4] is a dataset used in other studies into autonomous detection of spacecraft anomalies (i.e. [9], an LSTM-based study, and [56], a CNN-based approach). This dataset consists of curated telemetry streams from NASA’s Soil Moisture Active Passive (SMAP) [13] and Mars Science Laboratory “Curiosity rover” (MSL) [14] missions. We selected this as the dataset for our study because it offers a good baseline against which to compare our results.

### Table 2. SMAP/MSL Dataset Statistics, from [4]

| Statistic | SMAP | MSL | Total |
|---|---:|---:|---:|
| Total anomaly sequences | 69 | 36 | 105 |
| Point anomalies | 43 | 19 | 62 |
| Contextual anomalies | 26 | 17 | 43 |
| Unique telemetry channels | 55 | 27 | 82 |
| Input dimensions | 25 | 55 | - |
| Telemetry values evaluated | 429,735 | 66,709 | 496,444 |

The data in [4] has been scaled to between `(-1, 1)` and anonymised. “Model input data also includes one-hot encoded information about commands that were sent or received by specific spacecraft modules in a given time window.” [4]. This results in a collection of 82 multivariate data sets, with around 100 labelled anomalies in total across all data sets, as detailed in Table 2. Each telemetry channel is a multivariate time series of one target parameter and additional parameters to be used as contextual information. The target parameter is the time series to be forecast, in which anomalies are to be detected.

The data was pre-split by [4] into “train” anomaly-free data to establish the nominal conditions and “test” sets, one per telemetry channel, which contain the labelled anomalies. We used the same split as in the original study in order to have comparable results.

### Data Clustering

Initial inspection of the telemetry channels showed that different telemetry channels had varying general characteristics such as “spiky” or “flat”. We wanted to investigate the link between the characteristics of the telemetry channels and the best performing deep learning model architecture, and whether specific architectures work better for certain types of data. Manual classification is not feasible due to the number of telemetry channels, so our idea was to use an unsupervised clustering approach.

To associate the telemetry channels into clusters, we used an unsupervised clustering approach. Each class represents a particular set of characteristics. The method used the standard central moments (mean, standard deviation, skewness, and kurtosis) calculated for the target parameter of each telemetry channel using SciPy [57]. NaN values are set to 0. Therefore each telemetry channel was represented by a single four-dimensional vector. We applied K-Means clustering [58] to these four-dimensional vectors.

The handling of NaN values is required for the statistics skewness and kurtosis because some telemetry channels contain parameter values with no variance (“flat”, in Table 5); these are forced to zeros. Skewness measures the asymmetry of the probability distribution. For a constant data series, skewness is not defined, as skewness presupposes that there is variance in the data. Kurtosis measures the “tailedness” of the distribution. For a constant series, like skewness, kurtosis is also not defined because kurtosis measures the outliers and a constant series has none. Mean and standard deviation are defined in case of constant value so do not need to be treated for NaNs. Skewness and kurtosis are calculable for non-flat telemetry channels and give a better summary of the data than mean and standard deviation alone.

Our clustering focuses on the training data set, without anomalies, so as to identify what the “normal” behaviour of the parameter is, as summarised by the shape of the curve. In spacecraft operations this is the more likely scenario as often data channels have yet to experience an anomaly [9], [5].

#### Listing 1. Time Series Clustering Pseudo-Code

```text
For each telemetry channel i:
    Extract target parameter p_i from i
    Calculate central moments of p_i:
        [mean, standard deviation, skew, kurtosis] => vector_i
    Set any value (vector_i_j = NaN) => 0
    Add vector_i to list
Apply K-Means to list => n_clusters
```

The “elbow method” [59] is a heuristic to find an optimal number of clusters by looking for a change in slope. For the [4] data, the method indicated that 5 clusters of data types would be an optimal solution. The change in slope at k = 5 is clear. Distortion is an indication of how well the clusters fit, and k is the number of clusters. Lower values of k would suggest insufficiently separated clusters, whereas greater values would indicate overly split clusters. This result is dataset-specific, and may not reflect all spacecraft telemetry channels; however, the K-means method is portable to other data sets and fast: all 82 channels were processed in under a second.

The resulting clusters can be described as the following “types” of telemetry data according to the behaviour of the target data channel:

Cluster 0 “Binary”: the values alternate between one of two values. When scaled to (-1, 1), this shows as large spikes across the full range. There are 43 data channels in this cluster.
Cluster 1 “Flat”: the value is not expected to change at all. There are 21 data channels in this cluster.
Cluster 2 “Oscillating”: similar to flat, but the value oscillates around a certain value rather than being fixed. There are 11 data channels in this cluster.
Cluster 3 “Spiky”: occasional large changes in the data are expected and normal. These represent a particular challenge for univariate models as the cause of the spike can only be determined from additional data. There are 2 data channels in this cluster.
Cluster 4 “Complex”: combination of the other data types. There are 2 data channels in this cluster.

In addition to reporting the results per model architecture trained on all telemetry channels, the best results per (model architecture, cluster) combination will also be given. This will inform whether certain architectures work better on certain types of data (cluster), or whether there is a “one-size-fits-all” universal solution which is applicable to all types of data behaviour.

## 4. Experiments and Results

The results of the 13 different models (described in Section 2) show considerable difference in training times, ranging from a few hours to over one full day, as shown in Table 3. However, the performance of the models does not scale with processing time. The performance is measured by two key metrics: the F1 (%) score considering the number of anomalies correctly detected (F1 anomaly), and the F1 score of the number of time points correctly labelled as occurring within anomalies (F1 time point).

Table 3 shows, per model architecture implementation, the total training time and the average training time per telemetry channel. True positive (TP), false positive (FP), and false negative (FN) values are also given per anomaly.

It is expected that “F1 time point” will not be very high, as the nature of the threshold-based anomaly detector means that data points either side of a labelled anomaly may not be detected as anomalous themselves, even though a domain expert would label them as such. Nevertheless, it gives an indication of the overall model performance when determining if any given data point is anomalous. This is the metric used in [4] and we retain it to allow direct comparison of results between their study and ours.

The F1 score pertaining to the detected anomalies (“F1 anomaly”) is more significant in terms of perceived anomaly detection performance by the spacecraft operator [4], [9].

### Model Performance

The best performing model architecture in our study is the CNN-based XceptionTimePlus implementation, with F1 anomaly score of 69.9%. This is lower than the tuned results from the Telemanom study (Table 1) but represents a 6% better performance than the worst performing model here, FCNPlus. It is noteworthy that the best and worst performing models are both of the CNN architecture family (XceptionTimePlus 69.9%, FCNPlus 63.2%). This suggests that there is no intrinsic advantage of CNN-based models in general.

The hybrid TransformerLSTMPlus (69.6%) and RNN-based GRUPlus (69.1%) show similar performance although with vastly different training times.

Given the overall good performance of XceptionTimePlus (69.9%) and the relatively low training time, this architecture would be our recommendation for a general-purpose anomaly detector, as an initial investigation, before extensive effort is applied to tuning.

### Training Performance

Spacecraft parameters number in the tens to hundreds of thousands per spacecraft, so the effectiveness of the models in terms of training is of critical importance. Training a large number of slow-to-train models is thus potentially prohibitive, and models would need regular retraining as the spacecraft ages or environment changes. We introduce an F1 per second (F1/s) metric to provide a comprehensive view of our models’ performance, taking into account both their accuracy and computational efficiency. In essence, it indicates the F1 score achieved for every second of training. A higher value implies that the model not only performs well but can be trained quickly, making it both effective and efficient.

The CNN-based models ResNetPlus and XceptionTimePlus are the most efficient, whereas the “large” (many model parameters) CNN-based InceptionTimeXLPlus and RNN/Transformer hybrid LSTMAttentionPlus offer the worst performance/training time trade-off. Generally the Transformer-based architectures all suffer from a performance/training time trade-off. The difference in training times is stark—from 2.5 minutes for XceptionTimePlus to over 18.5 minutes for TransformerLSTMPlus, whereas their F1 scores are nearly identical (69.9% vs 69.6%).

### Best Architecture per Cluster

A further level of analysis into the results provides insight into the best performing models per type of telemetry data, as determined by the clusters shown earlier. To calculate the F1 anomaly scores per time point and per labelled anomaly, the true positives, false positives, and false negatives were summed across the relevant cluster.

All (architecture, cluster) pairs were ranked by F1 anomaly score to determine the best performing model for each cluster. The architectures identified as best and second-best performing for each cluster are given in Table 4.

It is notable that the best performing architecture identified in Table 3 (XceptionTimePlus, 69.9%) is not the best in each data type cluster. The CNN-based MultiInceptionTimePlus and XceptionTimePlus models perform best in the Binary (65.2%) and Flat/Oscillating clusters (86.3%, 72.0%) respectively, suggesting that the spatial awareness of the models is particularly useful in identifying anomalies in these cases.

With fewer telemetry channels, two apiece, the Spiky and Complex clusters are more difficult to assess in general terms due to the low number of examples in the data set. Despite being an older architecture, the MLP-based gMLP performs best for the Spiky (100%) and Complex (100%) cases, outperforming all other architectures. More examples of these clusters are needed before a recommendation can be made on these specific telemetry data types, but it is instructive to note the differences in performance. With an average F1 score of 84.7%, the best performing models per cluster collectively outperform any single model by nearly 15% absolute.

As an ensemble approach to the general anomaly detection problem, taking the best performing architecture per data type greatly increases the performance overall. The average F1 score of 84.7% surpasses the 83.6% achieved by the baseline study, Telemanom [4].

### Table 3. Results per Architecture

| Architecture | Total Time (d hh:mm:ss) | Avg Time / Channel (mm:ss) | F1 (%) Time Point | TP Anomaly | FP Anomaly | FN Anomaly | F1 (%) Anomaly |
|--------------|-------------------------|----------------------------|-------------------|------------|------------|------------|----------------|
| XceptionTimePlus | 00 03:34:27 | 02:39 | 37.42 | 65 | 16 | 40 | 69.89 |
| TransformerLSTMPlus | 01 00:58:58 | 18:30 | 36.69 | 64 | 15 | 41 | 69.57 |
| GRUPlus | 00 03:32:16 | 02:37 | 32.71 | 66 | 20 | 39 | 69.11 |
| ResNetPlus | 00 03:02:39 | 02:15 | 38.36 | 63 | 15 | 42 | 68.85 |
| MultiInceptionTimePlus | 00 05:32:11 | 04:06 | 36.71 | 60 | 10 | 45 | 68.57 |
| LSTM_FCNPlus | 00 23:15:57 | 17:14 | 32.83 | 66 | 23 | 39 | 68.04 |
| LSTMPlus | 01 02:21:02 | 09:46 | 35.10 | 67 | 25 | 38 | 68.02 |
| mWDNPlus | 00 05:53:12 | 04:22 | 34.99 | 63 | 21 | 42 | 66.67 |
| gMLP | 00 16:06:24 | 11:56 | 32.44 | 63 | 21 | 42 | 66.67 |
| TSTPlus | 00 13:29:22 | 10:00 | 34.94 | 57 | 16 | 48 | 64.04 |
| InceptionTimeXLPlus | 01 05:38:24 | 21:57 | 33.32 | 64 | 31 | 41 | 64.00 |
| LSTMAttentionPlus | 01 05:41:37 | 22:00 | 35.35 | 62 | 29 | 43 | 63.27 |
| FCNPlus | 01 00:57:41 | 09:15 | 36.29 | 61 | 27 | 44 | 63.21 |

### Table 4. Results per Cluster

| Architecture | Cluster | F1 (%) Time Point | F1 (%) Anomaly |
|--------------|---------|-------------------|----------------|
| MultiInceptionTimePlus | Binary | 31.65 | 65.22 |
| ResNetPlus | Binary | 34.02 | 64.58 |
| gMLP | Complex | 43.24 | 100.00 |
| LSTM_FCNPlus | Complex | 42.87 | 100.00 |
| XceptionTimePlus | Flat | 44.10 | 86.27 |
| TransformerLSTMPlus | Flat | 41.00 | 85.19 |
| XceptionTimePlus | Oscillating | 27.50 | 72.00 |
| ResNetPlus | Oscillating | 26.87 | 64.00 |
| gMLP | Spiky | 48.87 | 100.00 |
| MultiInceptionTimePlus | Spiky | 35.08 | 85.71 |
| **Average** | — | — | **84.7** |

## 5. Conclusion and Future Work

In conclusion, the insights derived from our study have shown innovative advancements in spacecraft anomaly detection, laying a robust foundation for future explorations and discoveries in this domain.

### Conclusion

In this work, we performed a comparative study of diverse deep learning model architectures, with the goal of assessing their efficacy in spacecraft anomaly detection. Our findings revealed that model XceptionTimePlus (69.9%) exhibited the most optimal performance among all the models assessed in the study, across all telemetry channels. However, it is important to note that the overall performance was not on par with the outcomes demonstrated in [4]. A contributing factor to this is the conscious decision to refrain from hyperparameter optimisation in order to preserve the default comparisons and allow direct relative comparisons. Nevertheless, our study provides valuable insights into which families of deep learning architecture perform well, and which do not.

Furthermore, due to constraints in computational resources it was not possible to follow the standard optimisation strategies such as grid search, which runs many iterations of the model to explore the hyperparameter space. With some models taking several days to run once (e.g. LSTMAttentionPlus at 1 day and 5 hours), it is infeasible to run the large number of iterations required.

In addition to this, our research illuminated that different deep learning model architectures exhibit varying degrees of proficiency depending on the nature of the data, be it “spiky”, “flat”, “complex”, “oscillating”, or “binary”. We introduced an innovative clustering methodology in this paper, facilitating the efficient allocation of spacecraft telemetry channels into distinct clusters contingent on the inherent statistical properties of the data, based on the shape of the curve. This novel approach has not only advanced our understanding but has also paved the way for the advent of more sophisticated ensemble models, based on individual models that are harmoniously optimized for disparate data types. This ensemble approach was able to exceed the performance of the baseline study (84.7% vs 83.6%), despite using unoptimised models.

### Future Work

The work in our study has suggested new possibilities and directions for future research. A natural extension of this work would be the exploration of ensemble models that are proficiently optimised to accommodate various data types, leveraging the clustering methodology introduced in this paper. Furthermore, a meticulous exploration of hyperparameter space will be pivotal to harness the maximal potential of the models, thereby advancing the state-of-the-art in spacecraft anomaly detection.

As described above, the individual models were not individually optimised per model, rather used defaults from the respective frameworks fastai and tsai. The success of the clustering approach suggests itself as an alternative approach to the one-model-for-all approach seen in other studies ([9], [4], [3]): that of creating a set of optimised hyperparameters per data type (spiky, binary, etc).

Additionally, current anomaly detection approaches (e.g. [3], [4], [5], [9]) rely predominantly on forecasting models to deduce nominal behavior, identifying anomalies through a comparative analysis of predictions against predetermined thresholds. A promising avenue for future research would be the application of deep learning classification techniques, which could potentially offer a direct assessment of the telemetry channels without relying on thresholds.

## BibTeX (optional)

```bibtex
@misc{lakey2024comparison,
  title={A Comparison of Deep Learning Architectures for Spacecraft Anomaly Detection},
  author={Lakey, Daniel and Schlippe, Tim},
  year={2024},
  eprint={2403.12864},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
