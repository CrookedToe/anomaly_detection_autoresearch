---
id: esa-adb-benchmark-2024
type: preprint
title: "European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry"
authors: "Kotowski, Krzysztof; Haskamp, Christoph; Andrzejewski, Jacek; Ruszczak, Bogdan; Nalepa, Jakub; Lakey, Daniel; Collins, Peter; Kolmas, Aybike; Bartesaghi, Mauro; Martinez-Heras, Jose; De Canio, Gabriele"
year: 2024
venue: "arXiv"
doi: "10.48550/arXiv.2406.17826"
url: "https://arxiv.org/abs/2406.17826"
arxiv: "2406.17826"
tags:
  - esa-adb
  - benchmark
  - satellite-telemetry
  - evaluation
added: 2026-03-26
source: manual
---

# European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry

## Abstract


## Summary

- Benchmark paper for **ESA-ADB**; defines the dataset and hierarchical evaluation pipeline used in this project’s Mission 1 subset and metrics (e.g. event-wise and channel-aware scoring).

## Full text (optional)

Open access: [arXiv:2406.17826](https://arxiv.org/abs/2406.17826).

## 1. Main

Monitoring satellite telemetry time series data for anomalies is a daily practice of thousands of
spacecraft operations engineers (SOEs) in mission control centres around the world. It ensures
the safe and uninterrupted operation of multiple scientific, communication, observation, and
navigation satellites. SOEs are typically supported with simple automatic anomaly detection
systems that alarm when a measurement goes outside its predefined nominal limits or when a
measurement correlates with a known anomalous pattern1,2, but more sophisticated anomalies
are usually detected manually which is a very expensive and error-prone task^3. For this reason,
all major space-related entities have been actively researching, developing and testing advanced
automatic anomaly detection systems in the past years, including European Space Agency
(ESA)4,5, National Aeronautics and Space Administration (NASA)^2 , Centre National d’Études
Spatiales (CNES)^6 , German Aerospace Center (DLR)^7 , Japan Aerospace Exploration Agency
(JAXA)^8 , and Airbus, among others^9 –^12. It is also one of the prioritised domains of Artificial
Intelligence for Automation (A2I) Roadmap^13 of ESA and there is a growing trend in applying
such systems directly on board satellites toward faster alarming and autonomous satellite health
monitoring^14.


There are hundreds of algorithms for time series anomaly detection (TSAD) proposed in the
literature (158 according to Schmidl et al.^15 ) that could be viable solutions for the space sector,
but currently, the main challenge everyone is facing regards the evaluation of different
approaches. This happens because there are relatively few anomalies in flying missions^3 and no
comprehensive data collection from multiple sources, thus it is hard to objectively conclude
that one approach works better than the other. Moreover, multiple recent papers show that the
majority of publicly available datasets, benchmarks, metrics, and protocols for TSAD are
flawed and cannot be used for an unbiased evaluation of emerging machine learning (ML)
techniques^16 –^19. In addition, real-life satellite telemetry is an especially challenging example of
a multi-variate time series with many specific problems and complexities related to its:

- high dimensionality and volume (years of recordings from up to thousands of channels
    per satellite^20 ),
- complex network of dependencies between channels,
- complex characteristics (i.e. varying sampling frequencies across time and channels;
    data gaps caused by idle states and communication problems; trends connected with
    the degradation of spacecraft components; concept drifts related to different operational
    modes and mission phases),
- diverse types of channels (i.e. large variety and ranges of physical measures, categorical
    status flags, counters, and binary telecommands),
- noise and measurement errors due to the influence of the space environment.

The European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry (ESA
Anomaly Detection Benchmark or ESA-ADB, in short) aims not only to address all the
mentioned challenges and flaws reported in the literature but also to establish a new standard
for ML-based satellite telemetry analysis and general TSAD. It consists of three main
components (visualised in Supplementary Fig. 1 for easier comprehension):

1. Large-scale, curated, structured, ML-ready ESA Anomalies Dataset (ESA-AD, in short)
    of real-life satellite telemetry collected from three ESA missions (out of which two are
    selected for benchmarking in ESA-ADB), manually annotated by SOEs and ML
    experts, and cross-verified using state-of-the-art algorithms.
2. Evaluation pipeline designed by ML experts for the practical needs of SOEs from the
    ESA’s European Space Operations Centre (ESOC). It introduces new metrics designed
    for satellite telemetry according to the latest advancements in TSAD16,18,21–^23 and


```
simulates real operational scenarios, e.g. different mission phases and real-time
monitoring.
```
3. Benchmarking results of TSAD algorithms selected and improved to comply with the
    space operations requirements.

The main goal of ESA-ADB is to allow researchers and practitioners to design and thoroughly
validate methods that could be directly applied in real space operations environments, taking
into account all real-life challenges of satellite telemetry.

## 2. Results

The detailed nomenclature used in ESA-ADB is explained in Supplementary Material 1.
Datasets and results are anonymised to avoid disclosing sensitive mission-specific information,
such as channel names, timelines, or types of measured values, among others. The
anonymisation does not affect the data integrity and it was verified that algorithms produce the
same results as before anonymisation, see Supplementary Material 2.3. It does prevent from
using physics-informed approaches^24 or domain-specific knowledge to design algorithms (for
example, to match telecommands and channels by names or to expect anomalies in specific
times, e.g. during increased solar activity). However, it also enforces the usage of universal
data-driven approaches, instead of focusing on mission-specific intricacies.

## ESA Anomalies Dataset

The summary statistics of two missions from ESA-AD are presented in Table 1. The third
mission from ESA-AD (Mission3) is not a part of ESA-ADB, because of a small number and
triviality of anomalies (according to Definition 1 of Wu & Keogh^18 ) and a large number of
communication gaps and invalid segments – see Supplementary Material 2.1. Hence, it is
omitted in this section for clarity. ESA-AD is publicly available under the link
https://doi.org/10.5281/zenodo..

The dataset includes 76 channels from Mission1 and 100 channels from Mission2, but only 58
and 47 channels, respectively, are monitored for anomalies (target channels) while the rest are
meant to support the detection process (non-target channels; see the detailed definition in
Supplementary Material 1.4). Channels are grouped into 6 subsystems – 4 in Mission1 and 5 in
Mission2, with 3 matching subsystems between missions. Additionally, related channels with


similar characteristics are organised into 18 (Mission1) and 29 (Mission2) numbered groups,
so it is easier to manage the dataset for ML purposes. For each mission, there are hundreds of
different telecommands with millions of executions, but only a small fraction directly relates to
annotated anomalies and rare nominal events. Although telecommands were initially prioritised
from 0 (least important) to 3 (most important), it is a part of the challenge to discover their true
importance for TSAD. The number of data points exceeds 700 million for each mission which
gives more than 7 gigabytes (GB) of compressed data in total. It is orders of magnitude more
than for any other public satellite telemetry dataset, especially NASA SMAP and MSL datasets^2
(see Supplementary Table 6 ). The number of points is proposed as the main measure of the
dataset volume because the duration of 17.5 years is not objective due to varying sampling rates
and anonymisation.

```
Table 1. Summary statistics of missions included in ESA-ADB.
Mission1^ Mission2^ Both missions^
Channels 76 100 176
Target / Non-target 58 / 18 47 / 53 105 / 71
Channel groups 18 29 47
Subsystems 4 5 6 *
Telecommands 698 123 821
Priority 0/1/2/3 345 / 323 / 19 / 11 0 / 0 / 119 / 4 345 / 323 / 138 / 15
Total executions 1,594,722 1,918,002 3,512,
Data points 774,856,895 776,734,364 1,551,591,
Duration (anonymised) 14 years 3.5 years 17.5 years
Compressed size [GB] 3.51 3.81 7.
Annotated points [%] 1. 80 0.5 8 1.1 9
Annotated events 200 644 844
Anomalies 118 31 148
Rare nominal events 78 613 690
Communication gaps 4 0 4
Univariate / Multivariate 32 / 16 4 9 / 63 5 41 / 79 9
Global / Local 113 / 8 3 585 / 59 698 / 142
Point / Subsequence 12 / 18 4 0 / 64 4 12 / 82 8
Distinct event classes 22 32 ** 54
* there are 3 matching subsystems between missions.
** including unknown anomalies as a single class.
```
The anomaly density, in terms of annotated data points, is between 0.57% (Mission2) and

1. 80 % (Mission1) which addresses the flaw of unrealistic anomaly density reported for many
popular TSAD datasets^18. There are 8 44 annotated events (anomalies, rare nominal events, and


communication gaps) in total. The majority of annotations for Mission2 are _rare nominal events_

- atypical but expected or planned changes in the telemetry that are not anomalies from the
operators’ point of view (e.g. commanded manoeuvres, resets, or calibrations), but are likely to
be detected as anomalies at their first occurrence by standard TSAD algorithms (see definitions
in Supplementary Table 1 ). It would be of high practical importance to design algorithms that
can recognise or memorise rare nominal events, so they are not alarmed as anomalies. There
are just 4 short communication gaps (missing data) reported for Mission1.

Each anomaly and rare nominal event is described by three attributes corresponding to its
dimensionality (uni-/multivariate), locality (local/global), and length (point/subsequence)
according to the adjusted nomenclature of anomaly types by Blázquez-García et al.^25. Most
annotations are categorised as multivariate global subsequence, but there is also a diverse set
of other types of anomalies (see Supplementary Table 7 for detailed statistics), including some
especially challenging ones (Supplementary Material 2.2). Additionally, similar events are
grouped into classes according to SOEs, so it is easier to analyse results and design detectors
targeted at a specific class. The distributions of classes of events across missions’ timelines are
presented in Fig. 1. Note that events of the same class can have different categories, e.g. resets
caused by telecommands are rare nominal events, but unexpected and non-commanded resets
are anomalies. This difference is also reflected by the subclasses of events.

Our dataset has several features distinguishing it from the majority of related datasets. It is
intended to reflect the raw telemetry data accessible for SOEs, with all its pros and cons,
including irregular timestamps, varying sampling rates, anomalies in training data,
communication gaps, and an overabundance of telecommands. Each channel has a separate set
of annotations (like in the latest SMD^26 , CATS^27 , and TELCO^28 datasets), because the same
anomaly may affect different channels in different ways and it is crucial to assess whether the
algorithms can properly indicate affected channels to operators. Additionally, a single annotated
event may be composed of multiple non-overlapping segments separated by nominal data, e.g.
a series of short attitude disturbances caused by the same anomaly. An example of such an
annotation is presented in Supplementary Fig. 9. This is to avoid assessing each segment as a
separate anomaly in the evaluation metrics (the unrealistic anomaly density flaw). Missions
vary significantly in terms of signal characteristics and specific challenges posed for TSAD
algorithms. They are summarised in Supplementary Table 2.


Fig. 1. **Distributions of events from different classes across timelines of Mission1 (top panel) and
Mission2 (bottom panel)**. The bar width corresponds to the event length, but for better visualisation,
the minimum width was limited to 10 and 2.5 days for Mission1 and Mission2, respectively. The
question mark represents anomalies of unknown class for Mission2.


## Benchmarking results

The objective of the benchmarking experiments is to validate the performance of selected
TSAD algorithms on ESA-AD using the proposed evaluation pipeline and the TimeEval
framework^29. The benchmarking code is publicly available under the link
https://github.com/kplabs-pl/ESA-ADB to ensure its full reproducibility. The results of this
study are intended to become a baseline and a starting point for future research. Hence,
experiments do not aim to extensively tune hyperparameters or to find the best algorithm for
satellite telemetry. They use default settings or parameters recommended by algorithms’
authors, sometimes adjusted to the specific features of our datasets (see Supplementary Material
3.4). This is done intentionally, to present the results of typical TSAD approaches and to
encourage the community to propose their own improvements and ideas. There are no
algorithms in ESA-ADB that can explicitly distinguish between anomalies and rare nominal
events, so the results in Table 2 are presented for all events (excluding only communication
gaps). However, separate results considering only anomalies are available in Supplementary
Table 9 for future comparisons. The metrics in the tables are ordered according to their priority
in our hierarchical evaluation pipeline. For F-scores, there are also corresponding precisions
and recalls for more detailed analysis. Unsupervised algorithms do not provide lists of affected
channels, so channel-aware and subsystem-aware scores are not reported. Scores are rounded
to 3 significant digits to account for the inherent uncertainty of manual annotations.

According to our hierarchical evaluation of the results in Table 2 , Telemanom-ESA-Pruned is
the best algorithm for Mission1. It achieves much higher corrected event-wise F0.5-scores than
any other algorithm, in all mission phases (Supplementary Table 13 ) and with a very high value
of 0.968 for anomalies only (Supplementary Table 9 ). It also achieves the highest alarming
precision thanks to its dynamic thresholding scheme (NDT) which merges adjacent detections
together. This highlights the importance of proper thresholding and postprocessing methods as
a part of an algorithm. On the other hand, pruning significantly decreases channel-aware,
subsystem-aware, and affiliation-based scores. Telemanom has the lowest ADTQC because 1)
being a forecasting-based algorithm, it tends to detect anomalies too early (low ADTQC _After
ratio_ ), and 2) the smoothing of forecasting errors applied in NDT strongly magnifies this effect.
Unsupervised algorithms perform very poorly for Mission1 in terms of event-wise scores. DC-
VAE-ESA and GlobalSTD are just slightly better which is especially disappointing for the
former deep learning method. The main problem of these algorithms is a massive number of
false detections caused by the noise and varying sampling rates in the data, as visible in the


examples in Supplementary Material 4. However, DC-VAE-ESA has the highest ADTQC and
affiliation-based scores, sometimes higher than for Telemanom-ESA. This suggests that more
advanced thresholding or postprocessing may significantly improve the event-wise scores.

For Mission2, simple Windowed iForest^30 and Telemanom-ESA-Pruned algorithms turned out
to be the best algorithms for the lightweight and full sets, respectively. Overall, unsupervised
algorithms perform relatively well for Mission2, sometimes better than deep learning-based
ones. It supports the need to always consider simple algorithms as a baseline18,31. Windowed
iForest achieved very high corrected event-wise F0.5-score (0.9 49 ), ADTQC (0.985), and
affiliation-based F0.5-score (0.9 59 ). The main reason is the relative triviality of the lightweight
subset of Mission2 which contains mainly rare nominal events characterised by significant
sudden changes in the signal (see Supplementary Fig. 5 ). However, the full set is much more
challenging and contains many less obvious events (see Supplementary Fig. 6 ). It is reflected
by much lower corrected event-wise F0.5-scores. Moreover, metrics for anomalies alone
(Supplementary Table 9 ) show that no algorithm was able to accurately identify 9 actual
anomalies in this overabundance of rare nominal events. This is the main challenge of this
mission. Mission2 is particularly problematic for Telemanom because of a lack of clear
periodicity and many commanded events that are impossible to forecast.

In most cases, the results in Table 2 for full sets of channels are much worse than for lightweight
subsets, but they are not directly comparable due to the lower number of annotated events in
the lightweight test sets (see tables in Supplementary Material 4.3). To allow for direct
comparison, Supplementary Table 10 presents the results of the DC-VAE-ESA and
Telemanom-ESA algorithms trained on full sets and tested on lightweight subsets. It confirms
the initial observation – event-wise precisions and F-scores for Telemanom-ESA are much
worse when trained on full sets of channels. This is one of the main challenges of high
dimensional telemetry data – the more target channels there are, the higher chance of false
detections is. Additionally, due to the strong interconnections between channels, false
detections frequently seep into many irrelevant channels. The similar comparison for DC-VAE-
ESA is inconclusive, but the overall results for this algorithm are relatively low.

Overall, the benchmarking results confirm that ESA-ADB poses a significant challenge for
typical TSAD algorithms and none of them offer a perfect solution for both missions, especially
for complete sets of channels. Some challenging events listed in Supplementary Material 2.
are not detected by any algorithm.


Table 2. **Benchmarking results for detection of all events (excluding communication gaps) in lightweight subsets of channels and all channels for
missions in ESA-ADB**. Boldfaced results indicate the best values among all algorithms (excluding _After ratio_ of ADTQC which is just a helper value).
**Mission1 – trained and tested on the lightweight subset of channels 4 1 - 46**
Metric PCC^32 HBOS^33 iForest^30 iForestWindow 30 KNN^34 Global STD3 Global STD5 ESA STD3DC-VAE- ESA STD5DC-VAE- TelemanESA- ESATeleman-Pruned-

```
Event-wise
```
```
Precision < 0.001 < 0.001 < 0.001 < 0.001 < 0.001 0.001 0.288 0.00 2 0.0 63 0.148 0.
Recall 0. 554 0.5 85 0.5 85 0.7 38 0.75 4 0.4 31 0.16 9 0.5 54 0.3 38 0.894 0.
F0.5 < 0.001 < 0.001 < 0.001 < 0.001 < 0.001 0.001 0.25 3 0.00 3 0.0 75 0.178 0.
Channel-
aware
```
```
Precision
Not available
```
```
0.4 31 0.16 9 0.5 50 0.3 38 0.894 0.
Recall 0.28 5 0.15 9 0.4 63 0.2 21 0.738 0.
F0.5 0.3 51 0.16 7 0.5 14 0.2 83 0.837 0.
Alarming precision 0.03 3 0.04 7 0.01 7 0.01 5 0.017 0.057 0.035 0.0 70 0.0 28 0.868 0.
ADTQC After ratScoreio^ 0.80.8^3340 0.760.7 813 0.0.78^7114 0.30.5^7563 0.60.80^123 0.9290.770^ 0.9090.688^ 0.9 0. 901^72 0.950. 8035 0.1360.428^ 0.1430.^
Affiliation-
based
```
```
Precision 0.5 35 0.54 3 0.543 0.59 9 0.522 0.559 0.69 9 0.5 84 0.7 80 0.727 0.
Recall 0.3 34 0.35 2 0.35 7 0.42 4 0.32 2 0.37 5 0.4 22 0.37 7 0.5 93 0.662 0.
F0.5 0.4 77 0.490 0.492 0.553 0.4 64 0.5 09 0.61 8 0.52 6 0. 734 0.713 0.
Mission1 – trained and tested on the full set of channels
Metric PCC^32 HBOS^33 iForest^30 iForestWindow 30 KNN^34 Global STD3 Global STD5 ESA STD3DC-VAE- ESA STD5DC-VAE- TelemanESA- ESATeleman-Pruned-
```
```
Event-wise
```
```
Precision < 0.001 < 0.001 < 0.
```
```
Out-of-
memory
```
```
Out-of-
memory
```
```
< 0.001 0.002 < 0.001 0.005 0.007 0.
Recall 0.870 0.957 0.9 01 0.848 0.761 0.924 0.804 0.946 0.
F0.5 < 0.001 < 0.001 < 0.001 < 0.001 0.003 < 0.001 0.007 0.008 0.
Subsystem
```
- aware

```
Precision
Not available
```
0. 520 **0. 728** 0. 526 0. 640 0. 676 0.3 95
Recall 0. 694 0 .538 0. 764 0. 670 0.8 59 **0.8 61**
F0.5 0. 528 0. 664 0. 538 0. 623 **0. 689** 0. 436
Channel-
aware

```
Precision
Not available
```
```
0.380 0.276 0.398 0.359 0.514 0.
Recall 0.292 0.208 0.414 0.266 0.569 0.
F0.5 0.325 0.241 0.350 0.282 0.477 0.
Alarming precision 0.003 0.002 0.001 0.004 0.049 0.002 0.017 0.074 0.
ADTQC After ratioScore^ 0.6130.642^ 0.4430.603^ 0.0.6^37843 0.7180.723^ 0.7430.691^ 0.647 0.752^ 0.7160.692^ 0.3220.673^ 0.4630.^
Affiliation-
based
```
```
Precision 0.563 0.539 0.5 59 0.560 0.575 0.559 0.578 0.545 0.
Recall 0.522 0.578 0. 503 0.492 0.462 0.476 0.511 0.368 0.
F0.5 0.554 0.547 0.5 47 0.545 0.548 0.540 0.563 0.497 0.
```

```
Mission2 – trained and tested on the lightweight subset of channels 1 8 - 28
Metric PCC^32 HBOS^33 iForest^30 iForestWindow 30 KNN^34 Global STD3 Global STD5 ESA STD3DC-VAE- ESA STD5DC-VAE- TelemanESA- ESATeleman-Pruned-
```
Event-wise

Precision 0.0 29 0.055 0.5 57 0.9 51 < 0.001 0.0 06 0. 061 0.003 0.064 0.188 **0.**
Recall **1.000** 0.91 1 0.97 4 0.94 0 **1.000 1.000 1.000 1.000 1.000** 0.986 0.
F0.5 0.0 36 0.068 0. 609 **0.9 49** 0.001 0.0 07 0. 075 0.003 0.079 0.224 0.
Channel-
aware

```
Precision
Not available
```
```
0.951 0.992 0.904 0.995 0.831 0.
Recall 0.462 0.372 0.554 0.451 0.870 0.
F0.5 0.767 0.723 0.787 0.783 0.822 0.
Alarming precision 0.06 1 0.105 0.075 0.21 7 0.0 60 0.05 4 0.06 1 0.052 0.068 0.912 0.
ADTQC After ratioScore^ 0.9 0. 99^83 9^ 0.9940.99 0 1.0000.99 1 0.90.985^48 0.3910.724^ 0.90. 99467 0.90.99^897 0.9080.996^ 0.9910.997^ 0.0870.507^ 0.3510.^
```
Affiliation-
based

```
Precision 0. 890 0.93 6 0.98 2 0.96 8 0.5 61 0.7 40 0.9 35 0.680 0.939 0.688 0.
Recall 0. 580 0.86 7 0.95 2 0.92 5 0.2 43 0. 296 0. 717 0.293 0.788 0.544 0.
F0.5 0.8 04 0.92 1 0.97 6 0.9 59 0.4 45 0. 569 0. 881 0.538 0.904 0.654 0.
Mission2 – trained and tested on the full set of channels
Metric PCC^32 HBOS^33 iForest^30 iForestWindow 30 KNN^34 Global STD3 Global STD5 ESA STD3DC-VAE- ESA STD5DC-VAE- TelemanESA- ESATeleman-Pruned-
```
Event-wise

```
Precision 0.0 64 0.016 0.02 2 0.
```
```
Out-of-
memory
```
0.0 06 0. 052 0.002 0.008 0.052 **0.**
Recall **0.9 97** 0.82 0 0. 903 0.74 6 **0.99 7** 0.9 92 **0.997** 0.994 0.992 0.
F0.5 0. 079 0.0 20 0.0 27 0.04 2 0.0 08 0. 064 0.002 0.011 0.064 **0.**
Subsystem

- aware

```
Precision
Not available
```
0.9 10 **0.9 83** 0.672 0.911 0.409 0.
Recall 0.9 65 0.9 51 0.967 0.952 **0.984** 0.
F0.5 0.91 0 **0.9 69** 0.699 0.907 0.451 0.
Channel-
aware

```
Precision
Not available
```
0. 899 **0.9 41** 0.774 0.931 0.584 0.
Recall 0. 550 0. 490 0.592 0.507 0.783 **0.**
F0.5 **0.7 91** 0.7 87 0.713 0.783 0.592 0.
Alarming precision 0. 094 0.1 48 0.1 12 0.1 79 0. 060 0. 077 0.066 0.083 0.771 **0.**
ADTQC After ratioScore^ 0.0.9^97590 0.0.93^9069 0.90.96^397 0.80.92^528 0.90.98^253 0.9 **0.99**^89 **3**^ 0.6630.825^ 0.9300.985^ 0.1040.513^ 0.2740.^

Affiliation-
based

```
Precision 0. 814 0.5 70 0.621 0.60 8 0. 680 0.9 04 0.603 0.859 0.586 0.
Recall 0.63 1 0.45 5 0. 499 0.4 74 0. 280 0. 628 0.324 0.625 0.348 0.
F0.5 0. 770 0.54 3 0.59 2 0.57 5 0. 529 0. 831 0.515 0.799 0.516 0.
```

The processing times of the algorithms are given in Supplementary Material 4.6. They are all
possible to run in real-time using our computational resources given in Supplementary Material
4.5.

## 3. Discussion

ESA-ADB is a starting point for the development of better algorithms for satellite telemetry
anomaly detection. It was designed in close collaboration between ML experts and SOEs to
fulfil the need for a reliable benchmark for both communities. The results show that ESA-ADB
poses a significant challenge for popular TSAD algorithms, and many changes had to be applied
in the TimeEval framework^29 , training procedures, and algorithms (i.e. Telemanom^2 and DC-
VAE^28 ) to make them applicable to our use case (i.e. to handle large datasets, tens of channels,
varying sampling rates, streaming evaluation, and anomalies in the training data). Although the
results of Telemanom-ESA-Pruned may seem promising, it is a highly parametrised approach
and the selected thresholds may not be optimal for other missions. It is an invitation for the
community to build upon those algorithms, try other ones (out of hundreds in the literature), or
propose new approaches to address the challenges and requirements of anomaly detection in
satellite telemetry.

The ESA Anomalies Dataset contains tens of target channels and millions of data points which
makes it a challenging data volume for most algorithms, while still being manageable using a
standard PC and relatively comprehensible for manual analysis. However, one needs to
remember that ESA-AD contains only a small subset of channels from selected missions. There
may be thousands of channels in actual telemetry and proposing a perfect solution for ESA-
ADB would be still just the first milestone on the way to reliable anomaly detection systems
for space operations. Moreover, potential solutions must be not only accurate but also fast
enough to be run in real-time on computational resources accessible to mission control and, as
the ultimate goal, on board satellites.

Our evaluation pipeline considers all recent recommendations for multivariate time series
anomaly detection16,18,22,23_._ It proposes new quantitative metrics, dataset splits simulating real
operational scenarios, and the idea of hierarchical evaluation. Some of the proposed metrics
may seem too strict when looking at the results, but they represent practical aspects of space
operations and encourage to look for better methods. Priorities or weights of aspects may differ
between use cases and only a part of the evaluation pipeline may be relevant in domains


different from satellite telemetry. Also, not every mission is appropriate for objective testing of
anomaly detection as shown in the example of the rejected Mission3. Our goal was to ensure
that improving the results of ESA-ADB does not create an illusion of progress but solves real
challenges in space operations and TSAD domains. To support this statement, Table 3 includes
a summary of how ESA-ADB addresses flaws reported by Wu & Keogh^18.

```
Table 3. A list of flaws reported by Wu & Keogh^18 and how they are addressed by ESA-ADB.
Flaw How does ESA-ADB address it?
```
```
Triviality
```
- ESA-AD is large and contains a diverse set of anomaly types and
    concept drifts which hamper the usage of simple algorithms
- ESA-AD offers a selection of non-trivial anomalies, so they can be
    evaluated separately (Supplementary Material 2.2)
- ESA-ADB includes a set of simple algorithms to verify the potential
    triviality of anomalies

```
Unrealistic
anomaly
density
```
- ESA-AD is large and the anomaly density in the dataset is below
    2% of data points
- There are only dozens of anomalous events per year
- Series of separate annotated segments within a short region are
    usually assigned to the same event and are treated as such when
    computing metrics
Mislabelled
ground truth
- While this flaw cannot be fully resolved in real-life datasets there
were several iterations of the annotation refinement process aided
by unsupervised and semi-supervised algorithms to identify
potential mislabelling^35
Run-to-
failure bias
- Anomalies are scattered across long, failure-free, operational
periods of acquired telemetry data from real satellite missions (see
Fig. 1 .)

ESA-ADB is an important departure point for further endeavours. The benchmark is meant to
evolve and new contributions by researchers and organisations to further improve and extend it
are welcome. Despite our best efforts, some mislabelling is inevitable in such substantial
amounts of real-life data, so we are open to requests for corrections, and we plan to release
updated versions of ESA-AD. Other potential improvements include adding new missions,
proposing new algorithms, and better thresholding schemes to fulfil all posed requirements. To
create even larger and better datasets, it would be very desirable to introduce a standardised
ML-oriented anomaly reporting system for space operations. Especially interesting algorithmic
directions are related to the adaptation of Matrix Profile methods36,37 and transformers with
positional time encoding^38. ESA-AD allows for testing few-shot and one-shot learning
techniques that utilise known anomalies in training sets and can memorise rare nominal
events^39. Due to its size and diversity, it can be useful in the domains of time series forecasting^40 ,


telemetry data compression^41 , and foundation models^42. Distribution shifts caused by some rare
nominal events make it an interesting resource in continual learning^43 and change-point
detection^44. The large collection of telecommands may be explored to test methods of decision
support for SOEs (one of use cases from the ESA A2I Roadmap^13 ).

ESA-ADB should allow researchers from academia and practitioners from space agencies and
space industry to cooperate, compete and develop newer approaches to solve a common
problem. Our future activities include the organisation of open data science competitions based
on our dataset, stimulating the international community to develop better methods for satellite
telemetry analysis. Enabling this exchange should help to produce useful solutions for the daily
practice of thousands of SOEs in mission control centres around the world and for future
autonomous space missions.

## 4. Methods

## Dataset collection and curation

Three missions (satellites) of different types (purposes, orbits, and launch dates) were selected
from the ESA portfolio based on the survey conducted among SOEs about the presence of
historical anomalies that are problematic to detect using existing out-of-limit approaches (some
of them are listed in Supplementary Table 4 ). The selection focused on collecting a large dataset
with a possibly diverse spectrum of signals and anomalies, to avoid common flaws of triviality,
unrealistic anomaly density, or run-to-failure bias^18. The most interesting continuous time
windows for anomaly detection were identified based on the occurrence of reported problematic
anomalies and other events reported in the Anomaly Report Tracking System (ARTS)
(artsops.esa.int/documentation/about) used at ESOC. Raw satellite telemetry was structured
according to Supplementary Material 2.4 and manually annotated in cooperation with SOEs
using the OXI annotation tool (oxi.kplabs.pl)^45 created specifically for the project needs. While
annotating, a special focus was put on the precise identification of anomaly starting points for
all channels. On the other hand, anomaly end times may be less accurate, because they are much
harder to identify objectively, especially for long anomalies. Importantly, ARTS reports are
intended for human use and are not well-suited for ML purposes. They usually include only
approximate time ranges and a small fraction of affected channels. Moreover, well-known
anomalies and rare nominal events are often not reported. Thus, the whole signal was carefully
revisited by the ML team in search of any suspicious events. An initial list of subsystems,


channels, and telecommands relevant for anomaly detection was proposed by SOEs, but was
gradually extended during several iterations of the annotation refinement process in which
overlooked anomalies were discovered by the ML team using different TSAD algorithms.
During this process, channels were divided into target and non-target for anomaly detection.
Non-target channels should only be used as additional information for the algorithms. They are
not annotated and are not assessed in the benchmark. Examples include status flags, counters,
and metadata such as location coordinates, where anomalies are not expected or it is not possible
to check for anomalies without external data. Related channels measuring the same physical
values and showing similar characteristics are organised into numbered groups, so it is easier
to manage the dataset for ML purposes, e.g. to train group-specific models or to visualise
results. For more details about data collection, annotation, and refinement processes, see our
previous related work (Kotowski et al., 2023)^35.

There are hundreds of different telecommands (TCs) in each mission. Some of them are critical
foe detecting annotated anomalies (i.e. when there is no reaction to the TC or the reaction is
different than usual) or distinguishing anomalies from rare nominal events. However, it may be
impractical to use them all in anomaly detection algorithms. Thus, 4 different priority levels for
TCs were introduced as a suggestion about their potential usefulness for anomaly detection
algorithms. The priorities from the least important to the most important are:

0. TCs not directly related to any subsystem included in the dataset.
1. TCs related to subsystems included in the dataset but not marked as potentially valuable
    for anomaly detection by SOEs.
2. TCs selected as potentially valuable for anomaly detection by SOEs.
3. A fraction of TCs of priority 2. assessed as valuable for anomaly detection by the ML
    team. The main rejection criteria were the scarcity of occurrences in the training data
    (less than 3) or no occurrences in the test data.

TCs of priority 3. are used as input for DC-VAE-ESA and Telemanom-ESA algorithms trained
on full sets of channels. These priorities are only suggestions and ESA-ADB users are welcome
to experiment with any combination of TCs.

```
Dataset division
```
Each mission is divided into halves of which the first half is taken as a training set and the
second half as a test set. This gives 84 months of training data for Mission1 and 21 months for


Mission2. In both cases, the last 3 months of the training set are taken as the validation set. As
agreed between the ML team and SOEs, 3 months is long enough to reliably monitor the
performance of algorithms in the latest environmental conditions. The validation and test sets
include only samples later than the training ones to avoid data leakage from future samples.
Anomalies appear in all sets, including training and validation ones. Such a division employs
all available data and it represents a mature phase of the mission in which a significant amount
of data is available for training. However, anomaly detection systems are also desirable for
SOEs already in the early phases of missions. Thus, shorter training sets are also proposed and
analysed in Supplementary Material 4.3 to assess the robustness of the algorithms to changing
mission conditions and to identify the earliest mission phase in which reliable detectors can be
trained.

**Lightweight subsets of channels**
In the default setting of ESA-ADB, all channels and telecommands (of priority 3.) are used as
input and all target channels are used as output from algorithms. However, anomaly detection
in tens or hundreds of channels simultaneously may be a very challenging task and it takes a lot
of computing power to process such data, so for initial experiments, familiarisation with ESA-
ADB, simpler models, and potential on-board applications, there are also lightweight subsets
of channels proposed in ESA-ADB. These are channels 4 1 - 46 from subsystem 5 for Mission
and channels 1 8 - 28 from subsystem 1 for Mission2. The selection is subjective, but the main
goal was to provide channels that are challenging for algorithms (in terms of the number and
difficulty of anomalies), interesting for SOEs (in terms of the satellite health monitoring),
relatively easy to visualise and analyse manually, and not strongly dependent on other channels
or subsystems (so they are possible to analyse in isolation from the whole system to some
extent). The lightweight subsets do not include any telecommands. Selected channels from
these subsets are presented in Supplementary Fig. 4 and Supplementary Fig. 9 for Mission1 and
in Supplementary Fig. 5 and Supplementary Fig. 8 for Mission2.

## Taxonomy of anomaly types

To the best of our knowledge, the taxonomy by Blázquez-García et al.^25 is the only one in the
literature that comprehensively defines multivariate anomaly types, and our definitions are built
based on this foundation. It divides anomaly types into point and subsequence ones, where point
anomalies are defined as single outlying data points. However, this definition does not take into


account varying sampling rates for which the length of “a single data point” may differ in time.
Thus, for our purposes, multi-instance point anomalies are allowed if they are relatively short
fragments of the signal that resemble points or peaks when inspected using a typical sampling
frequency for the channel. Both point and subsequence anomalies may be univariate or
multivariate depending on whether they affect one or more channels. Anomalies can
additionally be divided into global and local (contextual) ones, similarly as proposed in
behaviour-driven taxonomy by Lai et al.^46. To make the original definitions more specific in
our taxonomy, the global subsequence anomaly is defined as a subsequence of anomalous
values in which at least one instance can be treated as a global point anomaly.

In the proposed taxonomy, each anomaly type can be described by three attributes:
dimensionality (uni-/multi-variate), locality (local/global), and length (point/subsequence), as
presented in Supplementary Fig. 10. These attributes can be automatically inferred from per-
channel annotations:

1. **Dimensionality** can be inferred by counting the number of channels affected by an
    anomaly. One affected channel makes it univariate and more affected channels make it
    multivariate.
2. To infer **locality** , we calculate the minimum and maximum values of all nominal
    samples in the dataset for each channel. If any sample of an annotated event lays out of
    <min, max> range for any channel, we mark it as global, otherwise it is local. This
    approach is a bit simplistic taking into account severe distribution shifts and different
    nominal levels of the signal in some missions, but it should be enough to identify global
    anomalies which could be detected with an out-of-distribution approach from more
    challenging local anomalies.
3. In terms of **length** , considering non-uniform sampling rates and the differences between
    mission and channels, it is hard to give a strict definition of a point anomaly. One option
    is to make it dependent on the dominant sampling frequency for each mission (0.
    Hz for Mission1, 0.056 Hz for Mission2 and 0.065 Hz for Mission3). A point anomaly
    is defined as a sequence of up to 3 samples after resampling to the dominant sampling
    frequency. Importantly, some anomalies are fragmented into several non-overlapping
    annotated regions. In this case, we treat each region separately, so even if an anomaly
    contains several regions it can be a point anomaly if all of these regions are categorised
    as point anomalies.


Such automatically inferred attributes for every anomaly and rare event are given in
anomaly_types.csv for each mission, taking into account annotations for all channels. However,
when working with subsets of channels, only the specific subset of channels should be
considered to infer anomaly types. For this purpose, the script infer_anomaly_types.py is
available in the code repository. The attributes are not inferred for communication gaps and
invalid fragments.

## Metrics and hierarchical evaluation

The selection of metrics and evaluation pipeline is a crucial step in establishing a reliable
benchmark. Our selection is based on the close cooperation between SOEs and ML engineers
and is primarily targeted at practical aspects of mission control in ESA. Five such aspects were
identified and prioritised based on their importance for SOEs. They are listed in Table 4 together
with the metrics used to assess them. Importantly, each metric was designed to focus solely on
a single specific aspect, in the maximum isolation from the other factors. There are several
reasons for this, 1) to improve the interpretability of results by avoiding complex metrics
assessing multiple aspects at once, 2) to allow researchers from different domains to easily
reorder or discard priorities, and 3) to enable the hierarchical evaluation pipeline. In the
hierarchical evaluation pipeline, algorithms are compared for one aspect at a time, from the
highest to the lowest priority. The process continues to the next aspect only if the algorithms
are equal in terms of the previous aspect. This kind of evaluation has three important practical
advantages, 1 ) it puts a strong emphasis on the priorities suggested by SOEs, 2) there is no need
to select the weights of specific aspects, and 3) it saves computational time by calculating only
the necessary metrics.

The highest priority aspect relates to the proper identification of anomalous events, but with a
strong emphasis on avoiding false alarms at the same time (aspects 1a. “No false alarms” and
1b. “Anomaly existence” in Table 4 ). This is because false positives are costly to resolve and
deter operators from using the system. A high false positive rate is reported in the literature as
the main obstacle to the wider adoption of anomaly detection algorithms in space operations^2.
This fact additionally supports our idea of hierarchical evaluation, since a high false positive
rate disqualifies an algorithm even if it obtains perfect scores in other aspects. Moreover, many
other aspects focus only on performance for true positive detections (i.e. channel identification,
alarming precision, timing quality), so they indirectly depend on the anomaly existence aspect.


The second highest priority for SOEs is to have the information about subsystems (aspect 2a.
“Subsystems identification”) and channels (aspect 2b. “Channels identification“) affected by
anomalies. Proper subsystem identification is more important for SOEs as it gives a more
concise overview of the situation than a long list of specific affected channels. Again, it is of
paramount important to avoid false positives. It is strongly preferable to miss some channels
rather than to wrongly identify many irrelevant channels. ESA-AD contains tens of target
channels which is already hardly manageable for manual analysis, moreover, it is just a fraction
of channels from actual missions. Hence, an algorithm which does not provide affected
channels is of low practical utility, or even worse, it may amplify the “black box” nature of
advanced algorithms and decrease trust in this kind of system among operators. That is why it
was considered as a part of two _primary_ aspects of highest priority.

The following 3 _secondary_ aspects are not so crucial for SOEs but are certainly useful to
differentiate between algorithms having the same _primary_ scores. The 3rd priority is to avoid
algorithms that frequently repeat alarms for the same continuous anomaly segment (aspect 3.
“Exactly one detection per anomaly”). It is strongly connected to the highest priority (1a. “No
false alarms”), because even if one considered these repeated alarms as “true positives”, they
would be annoying and confusing to operators, nearly as badly as false positives. The last 2
priority levels directly relate to the anomaly detection timing. It is obviously better to detect
anomalies earlier than later (aspect 4. “Detection timing”), it is preferable to detect a whole
time range of an anomaly instead of just a part of it, and, in case of false detections, it is better
to show them close to real anomalies (aspect 5. “Anomaly range and proximity”). These aspects
are often highly emphasised in TSAD benchmarks from the literature, i.e. NAB^47 and
Exathlon^48. However, they are relatively less important for on-ground mission control.
Additionally, the latter aspect cannot be precisely assessed due to the mentioned problems with
the objective identification of some anomaly end times.

Despite many years of research in the domain, there is no consensus on a reliable and unified
set of TSAD metrics. Many recent advances criticise popular sample-wise and point-adjust
protocols for being overoptimistic and propose better alternatives16,17,19,21–23,49–^55. Some of these
latest recommendations are directly applied in ESA-ADB, i.e. the corrected event-wise F-
score^16 and affiliation-based F-score^22. Besides that, there are several constraints on the
selection of metrics arising from SOEs needs. Metrics must operate on binary detections, so
threshold-agnostic metrics based on continuous anomaly scores (for example, areas under
curves) cannot be used. Due to irregular timestamps and varying sampling rates, metrics must


operate in the time domain instead of the samples domain, so the evaluation is independent of
the algorithm-specific resampling. Each metric must be adapted to give a single score for
multivariate anomalies. The computational complexity of metrics calculation also matters when
dealing with large datasets. Based on that aspect, metrics with complexities higher than
quadratic such as VUS^53 are rejected, so the evaluation could run in a reasonable time.

The definitions of the proposed metrics are given in the following subsections. All
implementations are available in the published code. All metrics are defined in the <0, 1> range
where 1 is the perfect score. All metrics give an option to include only specific events in the
calculation. In default, only communication gaps are excluded, but in Supplementary Material
4.2 this feature is used to calculate results for anomalies only. Technical details of the
implementations are listed in the Supplementary Material 3.2.2.

```
Table 4. Priority aspects and proposed metrics for assessing algorithms in ESA-ADB.
Group Aspect with priority level and brief description Proposed metric
```
```
Primary
```
```
1a. No false alarms – minimise the number of
false detections
Corrected event-wise F0.5-score
1b. Anomaly existence – maximise the number
of correctly detected anomalies
2a. Subsystems identification – find a list of
affected subsystems Subsystem-aware F0.5-score^
2b. Channels identification – find a list of
affected channels Channel-aware F0.5-score^
```
```
Secondary
```
3. **Exactly one detection per anomaly** – avoid
multiple detections for the same annotated
segment

```
Event-wise alarming precision
```
4. **Detection timing** – determine the anomaly
start time as precisely as possible

```
Anomaly detection timing quality
curve (ADTQC)
```
5. **Anomaly range and proximity** – find the
exact duration of the anomaly and promote
detections in close proximity to the ground truth

```
Modified affiliation-based F0.5-score
```
```
Corrected event-wise F-score
```
Event-wise F-score promoted for satellite telemetry by Hundman et al.^2 has two features that
make it better suited for practical applications than the classic sample-wise (or time-wise)
approach, 1) all anomalies have the same weight independent of their length (in practice, short


anomalies may be even more important than long ones which are easier to spot manually), and
2) the metric value does not depend on a level of overlap of detections and ground truth (in
practice, it is usually enough to give an approximate location of the anomaly to human
operators). However, the classic event-wise precision has one serious flaw – an algorithm that
simply detects anomalies in every sample in the dataset would have a perfect score (see
Supplementary Fig. 11 ). To mitigate this, Sehili and Zhang^16 proposed to involve the true
negative rate (TNR) at the sample level (at the time level in our case) in the computation of the
event-wise precision. Such a corrected event-wise precision 𝑃𝑟𝑒𝑐𝑜𝑟𝑟 is defined by equation (1),

𝑃𝑟𝑒𝑐𝑜𝑟𝑟= (^) 𝑇𝑃𝑇𝑃𝑒
𝑒+𝐹𝑃𝑒

### ∙𝑇𝑁𝑅𝑡, 𝑇𝑁𝑅𝑡= 𝑇𝑁𝑁𝑡

```
𝑡
```
### , (1)

where 𝑇𝑃𝑒 is the number of event-wise true positives, 𝐹𝑃𝑒 is the number of event-wise false
positives, 𝑇𝑁𝑡 is the number of nanoseconds with true negatives, and 𝑁𝑡 is the number of
nominal nanoseconds. Based on that, the corrected event-wise 𝐹𝛽-score is defined by equation
(2):

```
𝐹𝛽𝑒𝑐𝑜𝑟𝑟=( 1 +𝛽^2 )(𝛽 2 𝑃𝑟∙𝑃𝑟𝑒𝑐𝑜𝑟𝑟∙^ 𝑅𝑒𝑐𝑒
𝑒𝑐𝑜𝑟𝑟)+𝑅𝑒𝑐𝑒
```
, 𝑅𝑒𝑐𝑒= (^) 𝑇𝑃𝑒𝑇+𝑃𝑒𝐹𝑁𝑒 (^) ( 2 )
The factor 𝛽 gives us the flexibility to control the relative importance of recall. Betas lower
than 1 are preferred in our case to weigh precision (fewer false positives) higher than recall
(fewer false negatives). It is challenging to objectively select a specific 𝛽, so following
Hundman et al.^2 the value of 0.5 was agreed as a good baseline. However, 𝛽 can be adjusted to
specific operational needs.
For multivariate anomalies, the metric is calculated between logical sums of annotations and
detections across all target channels. In rare cases where multiple events overlap in time, each
event is analysed separately, i.e. separate true positives (max. 1) and false negatives (max. 1)
are counted for each event.
**Subsystem-aware and channel-aware F-scores**
Typical TSAD metrics are applicable only in univariate settings with a single series of ground
truth annotations and detections as input. To get a single score for multiple channels, there must
be some aggregation performed, either across annotations/detections or across scores for
individual channels. Such aggregation loses information about the performance for individual


subsystems or channels, so it is impossible to assess their correct identification. In recent
articles56,57, special _anomaly diagnosis_ metrics are proposed to address this issue, namely
HitRate and Normalised Discounted Cumulative Gain (NDCG). These metrics measure how
relevant are the detected channels according to the list of annotated channels. However, they
need information about the relative relevance of detections which is not available when using
binary outputs. Thus, a new _anomaly diagnosis_ approach is proposed based on precisions and
recalls of identifying the list of affected subsystems and channels.

SOEs inspect potential anomaly sources at two levels of detail. First, they check which
subsystems are affected by the anomaly. Later on, they look at the specific channels affected in
those subsystems. The usefulness of algorithms supporting such inspection is proposed to be
measured with subsystem-aware (SA) and channels-aware (CA) F-scores. A subsystem is
counted as true positive 𝑇𝑃𝑆𝐴 if it has at least one annotated channel and at least one detected
channel (not necessarily the same) overlapping with the full time span of the anomaly (logical
sum of annotations across all channels in all subsystems). A subsystem is considered false
negative 𝐹𝑁𝑆𝐴 if it has at least one annotated channel but no such detections. A false positive
subsystem 𝐹𝑃𝑆𝐴 has no annotated channels but has at least one such detection. Thus, the
subsystem-aware F-score 𝐹𝛽𝑆𝐴 is given by equation (3):

```
𝐹𝛽𝑆𝐴=( 1 +𝛽^2 )(𝛽 2 𝑃𝑟∙𝑃𝑟𝑆𝐴𝑆𝐴∙^ 𝑅𝑒𝑐)+𝑅𝑒𝑐𝑆𝐴𝑆𝐴,
```
𝑃𝑟𝑆𝐴= (^) 𝑇𝑃𝑆𝐴𝑇𝑃+𝑆𝐴𝐹𝑃𝑆𝐴, 𝑅𝑒𝑐𝑆𝐴= (^) 𝑇𝑃𝑆𝐴𝑇+𝑃𝑆𝐴𝐹𝑁𝑆𝐴

### (3)

The channel-aware F-score 𝐹𝛽𝐶𝐴 is defined analogously, but an annotated channel is counted

as 𝑇𝑃𝐶𝐴 if it has any overlapping detection in the full time span of the anomaly. An annotated
channel is counted as 𝐹𝑁𝐶𝐴 if there is no such detection. A false positive channel 𝐹𝑃𝐶𝐴 has no
annotation but at least one such detection.

Again, 0.5 is used for 𝛽 as a baseline to be consistent with the event-wise F-score. In rare cases
where multiple events overlap in time, each event is analysed separately, i.e. separate true
positives (max. 1) and false negatives (max. 1) are counted for each event. Moreover, any false
positives related to correct detections of other overlapping anomalies are discarded, see
Supplementary Material 3.2.1 for a detailed example. For the lightweight subsets of channels
selected from a single subsystem, the subsystem-aware F-score is not reported.


```
Event-wise alarming precision
```
The corrected event-wise F-score counts only a single true positive even if there are multiple
separated detections for the same fragment in the ground truth (see Supplementary Fig. 12 ). In
practice, such redundant detections may be considered separate alarms which may be annoying
for operators. The event-wise alarming precision 𝑃𝑟𝐴 defined by equation ( 4 ) measures the ratio
of correctly detected events (𝑇𝑃𝑒) to the sum of correctly detected events and redundant alarms
(𝑇𝑃𝑟):

𝑃𝑟𝐴= (^) 𝑇𝑃𝑒𝑇+𝑃𝑒𝑇𝑃𝑟 ( 4 )
This metric may seem too strict in some cases, i.e. for many short detections very close to each
other, but it represents practical aspects of mission operations and encourages for applying
better thresholding or postprocessing approaches to avoid redundant alarms.
**Anomaly detection timing quality curve (ADTQC)**
The goal of this novel metric is to assess the accuracy of the anomaly start time identification
from the SOEs point of view. Some existing metrics of the anomaly detection latency, such as
the After-TP^21 or the Early Detection (ED)^58 , assume that an anomaly can be detected only
within its ground truth interval — _after_ it appears in the signal. However, the question arises
how to assess algorithms that detect anomalies too early — _before_ they start. They cannot be
assessed using After-TP or ED metrics but they certainly have some value. The Before-TP
metric^21 and the NAB score^47 rank earlier anomaly _predictions_ (to distinguish them from
_detections_ ) as better. However, in practice, as suggested by SOEs, too-early detections may be
seen as false positives by operators if they cannot confirm the existence of an anomaly within
a definable time. Thus, too early detections may decrease operators’ trust in an algorithm and,
in this context, are much worse than late detections of comparable distance from an anomaly
start time. According to SOEs, the quality of anomaly detection timing should decrease
exponentially for detections before the actual start time as opposed to much slower degradation
of quality for moderately late detections. A survey was conducted and confronted across SOEs
from different missions in ESA and KP Labs to define the timing quality in the range from 0 to
1 as a function of detection start time. The resulting consensus reflecting the operators’ point
of view is presented as the _anomaly detection timing quality curve (ADTQC)_ in Supplementary
Fig. 14 described by equation ( 5 ):


After agreeing on the shape of ADTQC, the most important issue was to select the operational
range of values for which the function should return a quality higher than 0, that is, for which
the detection is not useless from the practical point of view. The first straightforward step was
to define detections later than the anomaly end time (𝛽) as useless. Accordingly, detections
earlier than the anomaly length from the start time were also considered useless. Hence, the
shorter the anomaly the more accurately it must be detected to achieve similar quality value. In
the extreme case of point anomalies, ADTQC returns a value of 1 for exact detections and 0
otherwise. It makes sense from the practical point of view for two reasons, 1) detections for
short, hardly noticeable anomalies are likely to be considered false alarms if not well-timed,
and 2) end times of long anomalies are usually much harder to annotate precisely than for short
anomalies. Another unacceptable situation was identified when a detection is earlier than the
previous anomaly start time. When anomalies are close to each other, the detection timing must
be even more accurate to ensure their better separation.

The ADTQC metric value for the specific anomaly is determined by simply calculating the
value of the 𝐴𝐷𝑇𝑄𝐶(𝑥) function where 𝑥 is the difference between the detection start time and
the anomaly start time. Similarly to Before/After-TP, the metric is calculated and averaged
across all correctly detected events to get a final score in the range from 0 to 1. To support the
analysis of the results, the ratio of detections after the anomaly starting points to all detections
is calculated (called the _after ratio_ ).

For multivariate anomalies, the ADTQC metric is calculated between the logical sums of
annotations and detections across all target channels. It does not matter if the detections are for
correct channels because the metric focuses on the timing alone. The second possible approach
in the multivariate setting would be to calculate the ADTQC metric for each affected channel

```
𝐴𝐷𝑇𝑄𝐶(𝑥)=
```
```
{
```
```
0 , −∞<𝑥≤−𝛼
(𝑥+𝛼𝛼)
```
```
𝑒
, −𝛼<𝑥≤ 0
1
1 +(𝛽−𝑥𝑥)
```
```
𝑒,^0 <𝑥<𝛽
0 , 𝛽≤𝑥<+∞
```
```
, 𝛼,𝛽> 0
```
```
𝐴𝐷𝑇𝑄𝐶(𝑥)={^01 ,, 𝑥𝑥≠=^00 , (𝛼= 0 ∧𝑥≤ 0 )∨(𝛽= 0 ∧𝑥≥ 0 )
𝛼=min(𝑎𝑛𝑜𝑚𝑎𝑙𝑦 𝑙𝑒𝑛𝑔𝑡ℎ,𝑎𝑛𝑜𝑚𝑎𝑙𝑦 𝑠𝑡𝑎𝑟𝑡 𝑡𝑖𝑚𝑒−𝑝𝑟𝑒𝑣𝑖𝑜𝑢𝑠 𝑎𝑛𝑜𝑚𝑎𝑙𝑦 𝑠𝑡𝑎𝑟𝑡 𝑡𝑖𝑚𝑒)
𝛽=𝑎𝑛𝑜𝑚𝑎𝑙𝑦 𝑙𝑒𝑛𝑔𝑡ℎ
```
### (5)


separately. The average across all affected channels would be the final ADTQC score for a
specific anomaly. While this alternative approach would allow for more detailed quantification
of the anomaly detection timing across channels, it does not reflect the operators’ perspective
in which the first detection is the most important one, because it already enforces an action.
Later detections for any other channel do not matter so much, because operators are already
aware of the potential anomaly.

```
Modified affiliation-based F-score
```
The affiliation-based metric by Huet et al.^22 claims to resolve all the major flaws of previous
range-based metrics. That is, it is aware of the temporal adjacency of samples and anomalies
duration, has no parameters, and is locally and statistically interpretable (specific problematic
time ranges can be easily identified and a score of 0.5 means a random prediction). The main
idea is to divide the ground truth into local zones affiliated with consecutive anomaly ranges.
The borders of such _affiliation zones_ lie in the mid points between consecutive anomalies.
Precision and recall are calculated separately for each affiliation zone based on the average
directed distance between sets of annotated and detected points, either the distance from
annotated to detected (precision) or from detected to annotated (recall). This way it is easy to
analyse which zones are the most problematic for an algorithm. Affiliation-based F-score with
𝛽 of 0.5 is calculated to underscore the strong practical need to minimise the number of false
positives. The final global F-score is calculated as the arithmetic average of all local F-scores
(with each affiliation zone having the same weight).

An important modification to the original implementation relates to frequent situations when it
is impossible to calculate the precision in an affiliation zone (there is no detection, so there are
no true positives or false positives). In the original formulation, such an affiliation zone was
simply ignored when calculating an average precision over all affiliation zones. However, this
approach makes it hard to robustly compare different algorithms because of the different
numbers of affiliation zones taken into account, e.g. it gives a higher score to an algorithm that
detects a single anomaly very precisely and misses 4 others than to an algorithm that detects all
5 anomalies relatively well – see Supplementary Fig. 13. Thus, in our formulation, empty
detections get a precision of 0.5. An affiliation-based precision of 0.5 can be interpreted as a
random detection, so this modification promotes algorithms that would rather give an empty
detection than a false detection that is worse than random. There are also some other technical


adaptations to handle point anomalies and fragmented annotations, as described in
Supplementary Material 3.2.2.

## Preprocessing

Our dataset contains raw non-uniformly sampled timestamps, so only a few algorithms with
positional time encoding (such as TACTiS^38 ) could handle it without any resampling. The vast
majority of algorithms, including all the algorithms selected for ESA-ADB, operate only on
uniformly sampled time series. Additionally, there are many different types of channels,
including monotonic, categorical, and binary ones, so a consistent preprocessing procedure is
needed to run and compare the majority of algorithms.

```
Resampling
```
The vast majority of algorithms, including all the algorithms selected for ESA-ADB, operate
only on uniformly sampled time series. The zero-order hold interpolation (propagating the last
known value) is recommended for satellite telemetry in the OXI annotation tool^45. This
interpolation method is well suited for processing binary and quantised signals that are common
in satellite telemetry (i.e. telecommands and measurements from analog-to-digital converters)
because, unlike the linear or Fourier-based interpolation, it does not create any artificial,
impermissible values between points. More importantly, it does not use future samples to
perform the interpolation which is necessary in real-time applications. This interpolation is
presented in Supplementary Fig. 15 and implemented for the resampling as follows:

1. Construct a uniformly sampled list of timestamps in the target sampling frequency. Set
    the first/last timestamp in the list to the value of the earliest/latest original timestamp
    across all channels rounded down/up to the target sampling resolution. Fill the list
    between the first and the last element using uniformly sampled timestamps in the target
    frequency, e.g. if we resample a list of original timestamps <8:10:12, 8:10:14, 8:10:38>
    to the target frequency of 1/10 Hz (target resolution of 10 seconds), the resampled list
    will be <8:10:10, 8:10:20, 8:10:30, 8:10:40>.
2. Propagate the last known value and label from the original samples (zero-order hold) to
    each timestamp in the constructed list. If there are still any missing values for the initial
    element of the list (i.e. when some channels start a little earlier than others),


```
backpropagate the first known value from the original samples. This introduces a bit of
information from the future, but it usually concerns only a few samples at the beginning
of a test set.
```
3. Apply a correction for missing anomalies to ensure that no point events are removed
    due to the resampling. Iterate through consecutive pairs of unannotated timestamps in
    the resampled list and, if there are any annotated original points in between, take the last
    annotated sample and assign its value and label to the latter timestamp from the pair.
    The result of such a correction is visible in the rightmost sample of Channel_1 in
    Supplementary Fig. 15.

Target sampling frequencies differ across missions. The selection was based on the analysis of
the most densely sampled target channels, specifically, to prevent losing any annotated
anomalies, especially point anomalies:

1. In Mission1, 0. 033 Hz was selected based on the dominant sampling frequency of target
    channels 41 - 46 with some point anomalies.
2. In Mission2, 0.056 Hz was selected based on the dominant sampling frequency of all
    target channels. There are no point anomalies in Mission2, so there was no risk of losing
    point anomalies.
3. In Mission3, 0. 065 Hz was selected based on the dominant sampling frequency of all
    target channels.

```
Standardization
```
Standardization is a necessary step for some algorithms (such as KNN^34 ) and it may boost the
performance of neural networks^59. In our preprocessing, each channel is standardized separately
to zero mean and unit standard deviation according to nominal points in a training set after
resampling. However, such standardization is not performed for:

- algorithms that do not need it by definition, i.e. Isolation Forest^30 or COPOD^60 ,
- binary channels (any channel with only two unique values in the training data). Instead
    of being standardized, they are normalised to the <0, 1> range. These kinds of channels
    are quite common in satellite telemetry, i.e. telecommands or status flags. There are
    often just a few state changes, so the standard deviations may be very small and cause
    numerical errors,


- constant channels (with zero standard deviation). In this case, only the mean is
    subtracted,
- monotonic channels that are non-decreasing or non-increasing from the definition of
    the measured process, i.e. counters or cumulative on-times. In this case, the
    standardization is preceded by calculating the first difference of the resampled signal.

Channels with categorical values and status flags are enumerated according to the order of
occurrence of each state in the training set and standardized. This is a very naïve approach, but
it does not require laborious manual analysis of all channels and preparation of state mappings
for each potential mission. Also, it does not require special handling of categorical anomalies.
Moreover, categorical channels are usually non-target.

```
Telecommands’ encoding
```
TCs in the original data are represented by lists of timestamps at which specific TCs were
executed on board a satellite. For purposes of ESA-ADB, they are encoded as binary impulses
of a single sample length according to the target resampling resolution, so they are not removed
by the proposed resampling.

## Algorithms

There are several recent comprehensive reviews of approaches for TSAD15,17,18,25,49,54,61 that list
hundreds of TSAD algorithms. They can be divided into several groups according to the type
of learning (supervised, unsupervised, semi-supervised, weakly-supervised), the origin of a
method (classic machine learning, signal analysis, data mining, stochastic learning, outlier
detection, statistics, deep learning)^15 , supported dimensionality (univariate and multivariate),
and the main mechanism of anomaly detection (forecasting, reconstruction, encoding, distance-
based, distribution-based, decision trees, and rule-based systems)^15. It is technically infeasible
to implement and include all algorithms in ESA-ADB, so the selection had to be performed
based on substantive arguments.

The most fundamental and widely used approach for anomaly detection in spacecraft systems
is based on checking whether sensor values are within a predetermined nominal range. This
out-of-limits method has many advantages (i.e. simplicity, explainability, speed, minimal
computational requirements) and works well in surprisingly many situations. However, this


approach does not perform and scale well with the exponentially growing complexity of
spacecrafts (see Supplementary Fig. 2 ). The first data-driven machine learning approaches tried
to resolve this problem based on adaptive limit checking, principal components analysis (PCA),
and Bayesian networks^62. The Novelty Detection algorithm^63 implemented in the Mission
Utility and Support Tools (MUST)^1 of ESOC was one of the first proofs of concept that
demonstrated the possibility of having an efficient and generic telemetry monitoring system
based on machine learning. This initiative was noticed by the space operations community and
encouraged many entities to experiment with similar methods, including NASA^2 , CNES^6 ,
DLR^7 , JAXA^8 , and Airbus, among others^9 –^12. In recent years, an unprecedented success of deep
learning (DL) was observed in virtually all domains of science and industry, also encompassing
an array of space-related issues that relate to solving the problem of fuel-optimal landing^64 ,
designing the solar-sail trajectory for near-Earth asteroid exploration^65 , predicting risk of
satellite collisions^66 , and many more67,68. The notable DL-based algorithm designed for satellite
telemetry anomaly detection is the semi-supervised, forecasting-based Telemanom based on
recurrent neural network (RNN) with Long Short-Term Memory (LSTM) units which
established a baseline for all later related works, mainly because of the introduction of NASA
SMAP and MSL datasets. Many general-purpose TSAD algorithms have been validated on
those datasets, but their results are not indicative, because these datasets are widely criticised
in recent publications17,18,69.

Based on our review of existing TSAD frameworks and benchmarks, the TimeEval
framework^29 (github.com/TimeEval) was selected as the foundation for ESA-ADB. It offers
more than 70 implementations of TSAD algorithms of various types and a complete evaluation
pipeline. Its authors thoroughly tested it on real-life and simulated data^15. For purposes of ESA-
ADB, it was extended with several new algorithms (GlobalSTD, Telemanom-ESA, DC-VAE^28 ,
DC-VAE-ESA), metrics, and evaluation mechanisms. Importantly, default evaluation
procedures for unsupervised algorithms in TimeEval do not include any separate training step
on a training set. The algorithms are both trained and run on a whole dataset. This is a typical
setting for outlier detection tasks. However, this is not a realistic approach in online satellite
telemetry monitoring, and it would give an unfair advantage to unsupervised algorithms,
because of the information leakage from future samples. The framework and internal
implementations of some algorithms were modified, so each unsupervised algorithm is first
trained on the training set only (including calculating contamination levels, setting thresholds,


and standardization parameters) and then utilised to detect anomalies in the test set in an online
manner (without using future samples from the test set).

There are nine technical requirements for anomaly detection algorithms in satellite telemetry.
The first two are necessary to conform with the primary needs of SOEs (“ **shall** ”, following the
wording recommended by the European Cooperation for Space Standardization
ecss.nl/standard/ecss-e-st- 10 - 06c-technical-requirements-specification/) and the rest are
recommended in practical applications (“ **should** ”):

```
R1. Algorithm shall provide a binary response (i.e. 0 – nominal, 1 – anomaly). It is not
enough to provide continuous anomaly scores to SOEs, so a thresholding mechanism
should be a part of the algorithm. A clear boundary is needed to decide if something
should be alarmed to operators or not.
R2. Algorithm shall allow for real-time, online, streaming detection. Although on-ground
mission control usually does not work in actual real-time, because larger packets of data
are collected from a satellite only during infrequent communication windows, real-time
monitoring is desirable in the future of mission control and is necessary for on-board
anomaly detection systems.
R3. Algorithm should be able to model dependencies between multiple channels. Satellite
telemetry contains hundreds of interconnected channels and there are many examples
of anomalies that can be detected only when using information from multiple channels
at once.
R4. Algorithm should learn from anomalies in training and validation data.
R5. Algorithm should provide a list of channels affected by a detected anomaly. It is a
crucial aspect of practical applications in mission control.
R6. Algorithm should distinguish between target channels, non-target channels, and
telecommands, so it learns from all sources, but only anomalies in target channels are
reported to SOEs.
R7. Algorithm should learn to distinguish rare nominal events, so they are not alarmed after
the first occurrence.
R8. Algorithm should natively handle irregular timestamps and varying sampling rates,
without the need for additional resampling. Typical resampling schemes make
algorithms unaware of varying gap lengths between points which may lead to many
false anomaly detections.
```

```
R9. Algorithm should be possible to run in a reasonable time on a single high-end PC. The
specific limits are listed in Supplementary Material 4.5. The algorithm is not included
in our benchmarking results if they are not met, so it is effectively shall in our case.
```
Based on the initial requirements analysis, 20 algorithms were preselected among those
available (or added) in the TimeEval framework that at least partially fulfil all primary
requirements. Table 5 summarises the detailed requirements analysis for those algorithms.
Some examples of partially fulfilled requirements are for algorithms that R1) do not provide
dedicated thresholding mechanisms, R2) technically allow for the online detection but with a
large computational overhead, R4) handle anomalies in training data but cannot learn from
them, R 5 ) would need additional mechanisms or modifications of external libraries (i.e.,
PyOD^70 ) to provide a list of affected channels, R7) give only a theoretical option to learn rare
nominal events, or R9) are only possible to run for the lightweight subsets of channels (i.e.
Windowed iForest and KNN). None of the preselected algorithms are able to explicitly learn
rare nominal events (R7) or handle varying sampling rates (R8).

Based on the detailed analysis of the requirements, eight algorithms of various types were
selected for ESA-ADB, five unsupervised – principal components classifier (PCC)^32 ,
histogram-based outlier score (HBOS)^33 , isolation forest (iForest)^30 , k-nearest neighbours
(KNN)^34 , and three semi-supervised ones – global standard deviations from nominal
(GlobalSTD), Telemanom^2 , and DC-VAE^28. The selected unsupervised algorithms have several
important limitations in terms of TSAD. They may be give suboptimal results because of the
assumptions of independence of samples and identical fractions of anomalies in training and
test data (they fulfil R4 because they learn contamination levels from the training data). They
only give global scores, so it is impossible to calculate subsystem-aware and channel-aware
scores for them. They also do not support non-target channels and telecommands on input, so
this information was not used. However, they establish a baseline for more advanced
algorithms.

Among the rejected ones, Matrix Profile-based methods like DAMP^36 or MADRID^37 seem to
be promising candidates due to their outstanding speed, high interpretability, and a theoretical
possibility to memorise rare nominal events. However, they would need a special adaptation to
support multidimensional data^71 , they do not natively handle anomalies in training, and their
implementations in Matlab pose several technical and licensing problems when integrated with
TimeEval. The COPOD algorithm does not fulfil R9 after adapting it to online detection


required by R2. LOF^72 , k-Means^73 , Torsk^74 , and RobustPCA^75 showed very poor results in
initial experiments. All semi-supervised algorithms that only partially fulfil R9 were rejected.

```
Table 5. Analysis of preselected algorithms according to ESA-ADB requirements. 0/0.5/1 means
that the requirement is not/partially/fully fulfilled. Asterisks mark new methods added to the
TimeEval. Bold-faced requirements are “ shall ”.
Algorithm R1 R2 R3 R4 R5 R6 R7 R8 R9 Included in ESA-ADB
```
```
UNSUPERVISED
```
```
COPOD^60 1 0.5 1 1 1 0 0 0 1 NO
HBOS^33 1 1 0 1 0.5 0 0 0 1 YES
iForest^30 1 1 1 1 0.5 0 0 0 1 YES
Windowed iForest^30 1 1 1 1 0.5 0 0 0 0.5 SUBSETS
k-Means^73 1 1 1 1 0.5 0 0 0 0.5 NO
KNN^34 1 1 1 1 0.5 0 0.5 0 0.5 SUBSETS
LOF^72 1 1 1 1 0.5 0 0 0 0.5 NO
Matrix Profile36,37 1 1 0.5 0 0.5 0 0.5 0 1 NO
PCC^32 1 0.5 1 1 0.5 0 0 0 1 YES
Torsk^74 0.5 1 1 1 1 0 0 0 0.5 NO
```
```
SEMI
```
- SUPERVISED

```
DAE^76 0.5 1 1 0 1 0 0 0 0.5 NO
DC-VAE^28 * 0.5 1 1 0 1 0 0 0 0.5 NO
DC-VAE-ESA* 1 1 1 0.5 1 1 0 0 1 YES
GlobalSTD* 1 1 0 0.5 1 0 0 0 1 YES
Hybrid KNN^77 1 1 1 0 0 .5 0 0.5 0 0.5 NO
LSTM-AD^78 0.5 1 1 0 0 0 0 0 0.5 NO
OmniAnomaly^26 0.5 1 1 0 0.5 0 0 0 0.5 NO
RobustPCA^75 0.5 0.5 1 0.5 0.5 0 0 0 1 NO
Telemanom^2 1 1 1 0 1 0 0 0 0.5 NO
Telemanom-ESA* 1 1 1 0.5 1 1 0 0 1 YES
```
The published code contains implementations of all methods listed in Table 5. New and
improved algorithms introduced in ESA-ADB are described in the following subsections.

```
GlobalSTD
```
In this classic distribution-based approach, any samples deviating from the mean of the channel
by more than N its standard deviations are detected as anomalies. This approach is categorised
as semi-supervised because only nominal samples (excluding rare nominal events) from the


training set are used to compute means and standard deviations for each channel to avoid the
influence of outliers. In practice, the threshold of 3 standard deviations (STD3) is frequently
used (following the empirical statistical rule that 99.7% of data occurs within 3 standard
deviations from the mean within a normal distribution^79 ), but it may not be optimal when the
number of false positives should be minimised, so the threshold of 5 standard deviations (STD5)
is also tested to provide a versatile baseline for other algorithms. The main disadvantage of this
algorithm is that it is unable to detect local anomalies, so it is usually not a good choice in
practice. It is also not aware of dependencies between channels and it is very vulnerable to
changes in the data distribution during the mission. It also cannot use the information about
non-target channels and telecommands.

```
Telemanom-ESA
```
This semi-supervised algorithm proposed by NASA engineers^2 is an important point of
reference in the domain. It can be considered the most popular algorithm for anomaly detection
in satellite telemetry. Its core element is an LSTM-based RNN that learns to forecast a small
number of time points (10 by default) for a single channel based on the hundreds of preceding
samples (250 by default) from multiple input channels. The mean absolute difference between
the forecasted samples and the real signal is treated as an anomaly score, which is thresholded
using the non-parametric dynamic algorithm (NDT) to find anomalies. However, this “non-
parametric” approach (in the sense that it does not use Gaussian distribution parameters to
estimate thresholds) has several hyperparameters. In one of our previous works, a genetic
algorithm was used to find optimal hyperparameters of thresholding for NASA SMAP and MSL
datasets^80. However, this wrapper approach would be too computationally expensive to run on
our large datasets, so the default settings proposed by the authors are used.

Telemanom has several major issues that had to be addressed for the purposes of ESA-ADB.

- **Memory inefficiency** – Telemanom was designed for small and simplified datasets
    provided by NASA. Hence, the code is not optimised to handle very large datasets and
    it results in out-of-memory errors, e.g. there are many unnecessary copies of data, all
    training windows are loaded into memory at once, and binary annotations are loaded to
    memory as floating-point numbers. **Telemanom-ESA:** The code is optimised for
    memory consumption by using lazy generators to prepare training batches, in-place
    operations instead of copying data to new variables, and optimised data types.


- **Magic numbers in thresholding** – there are several conditions in the thresholding code
    that are not well documented in the original article. Especially impactful is that windows
    with smoothed errors below 0.05 are never anomalous
    (github.com/khundman/telemanom/blob/26831a05d47857e194a7725fd982d5dea5402
    dd4/telemanom/errors.py#L339). This is a very data-specific condition that is not well-
    suited for channels with certain signal values. **Telemanom-ESA** : This specific
    condition was removed from the code. **Telemanom-ESA-Pruned** : The threshold of
    0.05 is much too high for ESA-ADB, so it was changed to 0.007 based on the manual
    analysis of smoothed errors in the training data of both missions. This selection is highly
    subjective and is probably not optimal, but allows to assess the effect of such a pruning
    on the results.
- **No proper handling of anomalies in training data** – Telemanom assumes that there
    are no anomalies in the training set which is not true in our real-life setting.
    **Telemanom-ESA** : only continuous nominal parts longer than 260 samples and without
    any anomalies in any target channel are used for training and validation.
- **Only a single output from the LSTM model** – a single Telemanom model can take
    multiple input channels but it always outputs a prediction for a single target channel.
    This is a significant shortcoming when scaling this approach to hundreds of channels
    and gigabytes of data. The training of a single model may last hours or days, so training
    separate models for tens of channels can take months on a single PC. Also, it is
    impossible to provide different sets of input (non-target channels, telecommands) and
    output (target) channels. **Telemanom-ESA** : the output of Telemanom is extended, so
    that is possible to forecast any number of channels at once from a single model, like in
    DC-VAE^28. The channels are still analysed separately, but there is no need to train a
    separate model for each channel.
- **Problems with GPU support** – the original implementation of Telemanom is based on
    TensorFlow version 2.0 which does not natively support the CUDA compute capability
    8.6 of our Nvidia GPUs. Also, the TimeEval framework lacks GPU support.
    **Telemanom-ESA** : TensorFlow is upgraded to version 2.5 and the GPU support is added
    to the TimeEval.


### DC-VAE-ESA

DC-VAE (Dilated Convolutional – Variational Auto Encoder)^28 is one of the latest published
multivariate TSAD algorithms. It is a reconstruction-based method that relies on dilated
convolutions to capture long and short-term dependencies without using computation- and
memory-intensive multi-layer RNNs. Unlike the original Telemanom, it outputs multiple
channels from a single model and does not need a complicated thresholding scheme, because it
also estimates nominal standard deviations for each sample in each channel, so that thresholding
can simply be applied by looking for real samples exceeding reconstructions by more than N
standard deviations. In the original implementation, N is selected from integers between 2 and
7 to maximise the range-based F1-score^81 for each channel in the training set. This approach
does not scale well with the number of channels and assumes the similarity of anomalies
between the training and test sets. Thus, in DC-VAE-ESA, only two values of N are considered,
3 (STD3) and 5 (STD5). The DC-VAE paper introduces the TELCO dataset, which has three
rare features also promoted by ESA-ADB, i.e. separate annotations for each channel, anomalies
in training sets, and the idea of gradually increasing training set sizes. Hence, the modified DC-
VAE-ESA introduces only two small technical improvements to fully cover 7 of the 9
mentioned requirements:

- an option to handle different numbers of input and output channels,
- L2 regularisation of convolutional layers with the 0.001 rate to stabilise the training of
    VAE in the presence of concept drifts,

## 5. Data availability

The dataset is publicly available at https://doi.org/10.5281/zenodo.12528696 under CC BY 3.0
IGO license.

## 6. Code availability

The code is publicly available at https://github.com/kplabs-pl/ESA-ADB under the MIT
license.
## BibTeX (optional)

```bibtex
@misc{kotowski2024esa,
  title={European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry},
  author={Kotowski, Krzysztof and Haskamp, Christoph and Andrzejewski, Jacek and Ruszczak, Bogdan and Nalepa, Jakub and Lakey, Daniel and Collins, Peter and Kolmas, Aybike and Bartesaghi, Mauro and Martinez-Heras, Jose and De Canio, Gabriele},
  year={2024},
  eprint={2406.17826},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
