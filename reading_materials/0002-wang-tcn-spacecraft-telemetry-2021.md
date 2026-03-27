---
id: wang-tcn-spacecraft-telemetry-2021
type: paper
title: "Anomaly Detection of Spacecraft Telemetry Data Using Temporal Convolution Network"
authors: "Wang, Yuan; Wu, Yan; Yang, Qiong; Zhang, Jun"
year: 2021
venue: "IEEE International Instrumentation and Measurement Technology Conference (I2MTC)"
doi: "10.1109/I2MTC50364.2021.9459840"
url: "https://doi.org/10.1109/I2MTC50364.2021.9459840"
arxiv: ""
tags:
  - tcn
  - spacecraft-telemetry
  - anomaly-detection
added: 2026-03-26
source: manual
---

# Anomaly Detection of Spacecraft Telemetry Data Using Temporal Convolution Network

## Abstract


## Summary

- **TCN-based** anomaly detection aimed at **spacecraft telemetry**; useful direct precedent for architectural choices in `train.py` (temporal convolutions vs. RNN-style models).

## Full text (optional)

IEEE Xplore via the DOI above.

I. INTRODUCTION
Spacecraft is a complex system which makes great
contributions to meteorological observation, military
reconnaissance and resource survey, etc. However, with the
gradual improvement of the design complexity and
functionality of spacecraft, it is a challenge to process and
analyze data for safe and reliable operation of spacecrafts only
through traditional reliability engineering methods [1].
Therefore, it is necessary to monitor abnormal data which are
different from normal data or not in line with the spacecraft
working mode setting. Timely and effectively discover
abnormal patterns, and repair or troubleshoot in time remote
command repair, transmission link repair, which is important
for improving the quality of ground service, maintaining the
maturity, safety and reliability on the design and production
of the spacecraft have significant practical meaning. Thus, the
main purpose for anomaly detection is to detect the
components failed, evaluate health statues and warn in
advance.
```
```
For on-orbit spacecraft, in order to monitor the internal
operating status and obtain real-time data, the information is
measured by sensors in the spacecraft telemetry system, which
is converted into electrical signals. After the signals are
combined according to a certain rule, the radio
communication technology is used to transmit signals to the
ground telemetry equipment which includes receivers,
antennas, and split demodulators [1]. The ground equipment
restores the original variable information of each channel
through the signal demodulation technology. Those variables
obtained in this case is defined as the spacecraft telemetry data
[2]. With the increasement of society requirements, the
number of spacecrafts increases gradually so that the
dimension of telemetry data which needs real-time monitoring
expands extremely. It is a challenge to monitor massive data
by humanity and the experience of experts. Therefore,
analyzing the dataset is of benefit to the transition from
collecting, transmitting, storing data to mine the value of data.
Telemetry data are multidimensional time series [3]. The
anomaly detection for multivariate telemetry data draws much
attention on time series prediction methods in recent years.
```
```
At present, the time series prediction methods based on the
data-driven model [4, 5] are developed rapidly. The earliest
use of data-driven model is Autoregressive (AR) which acts
on linear time series, but there are many parameter restrictions
for stationary time series. The introduction of the differential
process does not require conditional restrictions to detect
weakly stationary time series data. Box and Jenkins et al. [6]
proposed Autoregressive Moving Average (ARMA) in 1920s.
In terms of strongly stationary time series, Box and Jenkins et
al. [7] proposed Autoregressive Integrated Moving Average
Model (ARIMA) in 1970s. From the evolution of statistical
models, the AR model makes a great contribution. As the
diversity of data features increases, machine learning occupies
the main position in prediction in recent years. Back-
Propagation (BP) is the classical algorithm in machine
learning. Zhao et al. [8] predicted toxic activity on different
datasets based on Support Vector Machine (SVM).
Rasmussen et al. [9] proposed Gaussian Process Regression
(GPR) which integrated statistics and machine learning. Both
SVM and GPR can support not only one-dimensional data, but
also multidimensional data. However, the above models do
not achieve a good performance with large amount of complex
data. Big data increase computation complexity and occupy
amount of resources. The propose of Recurrent Neural
Network (RNN) [10] effectively decreases the requirement of
computation resources, but traditional RNN often occurs
vanishing gradient or explosion when analyzing long time
series. Long Short Term Memory (LSTM) [11], as a special
RNN structure, is proposed to solve the above problems. Also,
LSTM reserves advantages of RNN and has the ability of
remembering long-term series. Due to the fact that LSTM
```
```
978-1-7281-9539-1/21/$31.00 ©2021 IEEE
```
```
This full text paper was peer-reviewed at the direction of IEEE Instrumentation and Measurement Society to the acceptance and publication.
```
2021 IEEE International Instrumentation and Measurement Technology Conference (I2MTC) | 978-1-7281-9539-1/21/$31.00 ©2021 IEEE | DOI: 10.1109/I2MTC50364.2021.

```
Authorized licensed use limited to: COLORADO STATE UNIVERSITY. Downloaded on March 26,2026 at 23:52:55 UTC from IEEE Xplore. Restrictions apply.
```

merely parses one batch of data, this model consumes long
time. Bai et al. [12] uses Convolution Neural Network (CNN)
on time series to gain parallel ability and large receptive field
which is the important factor for temporal characterization.
The proposed method is named as Temporal Convolution
Network (TCN). Large receptive field and parallel ability
greatly shorten the model inference time.

TCN is widely used in time series in recent years and has
been applied to anomaly detection. However, anomaly
detection of multidimensional telemetry data from spacecraft
based on TCN has not been researched. It is meaningful to do
such a research due to the complexity of the correlation
between spacecraft telemetry data. The rest of this paper is
organized as follows. Section II reviews the related
methodologies of TCN. Section III introduces the proposed
anomaly detection model. Section IV does an experiment with
NASA data by using the proposed algorithm, and then
analyzes the results from two aspects: the detection accuracy
and the model inference time. Section V concludes this paper
and introduces the future work.

II. RELATED METHODOLOGIES
TCN is a specific CNN which is suitable for time series
analysis. It includes four fundamental structures, namely one-
dimension (1-D) convolution, causal convolution, dilated
convolution and the residual connection [13].

_A. 1-D Convolution Network_

Under the multi-variable input, the input and output
usually have different size. Figure 1 intuitively illustrates how
1-D convolution network works with three kernel size. In
order to get the output, take the dot product of the input
subsequence and the kernel vector of the same length with the
learned weights. In order to make the visualization easier, the
following figures no longer show the dot product with the
kernel vector.

```
·
```
```
kernel size
```
```
· · dot product
```
```
input
```
```
output
```
Fig. 1. The illustration of 1-D convolution network.

_B. Causal Convolution_

TCN has two essential principles. One is that the size of
the input and output are the same. As for the constraint, zero-
padding can ensure each layer have the same length. Another
is to guarantee no future information leakage occurs. Causal
convolution can implement the requirement. In TCN, the
convolution operation is performed strictly in time sequence,

##### that is, the convolution operation at time t only occurs on

the data and the previous layer before the time _t_  1. The
illustration of causal convolution is shown in Fig. 2 with three
kernel size and two hidden layers. Suppose that the input
sequence is **X** [, , , ] _xx_ 12 _xT_

```
y
,], T , and convolution kernel is
```
##### defined as F [, , , ] ff 12 , ffKKK ], where K is the size of

convolution kernel. The causal convolution at _xT_ is

expressed as follows:

#### 

```
1
```
```
K
TkTKk
k
```
```
Fx fx 
```
(^) ¦ 
**input
hidden layer
hidden layer
output
x 1 x 2** Ċ **xT**
Fig. 2. The illustration of causal convolution.
_C. Dilated Convolution_
The modeling length of simple causal convolution on time
is limited by the size of convolution kernel, which is a problem
traditional CNN exists. Stacking layers linearly increases
receptive fields and generates long-time dependence. Dilated
convolution can reduce the depth of simple causal convolution.
Figure 3 shows the dilated convolution with the dilated factors
_d_ is equal to 1, 2 and 4 respectively between each layer. The
expression of the dilated convolution is as follows:
 (^)
1
_K
dT kTKkd
k
Fx fx_ 
(^) ¦ 
It is common to expand the dilated factors and increase the
kernel size to enhance receptive fields.
**input
hidden layer
hidden layer
output
x 1 x 2 xT
d = 1
d = 2
d = 4
· · ·**
Fig. 3. The illustration of causal dilated convolution network
_D.Residual Connection_
The residual connection is proved to be an effective
method for training deep networks, which enables the network
to transmit information in a cross-layer manner. The diagram
of the residual connection is given as Fig. 4. This paper
constructs one residual block with two-layer dilated causal
convolution and ReLU activation function which introduces
non-linearity. WeightNorm and dropout are used for
regularization. The output result of residual block can be
expressed as 
 _Fx_ () () _TdTT_  _F x x_  


```
Dilated Causal Conv
```
```
WeightNorm
```
```
ReLU
```
```
Dropout
```
```
Dilated Causal Conv
```
```
WeightNorm
```
```
ReLU
```
```
Dropout
```
```
+
```
```
Optional 1×
Conv
```
```
Output from previous
residual block
```
```
Residual block (k,d)
```
```
Input for next
residual block
```
Fig. 4. The diagram of residual connection

III. ANOMALY DETECTION BASED ON TCN TIME SERIES
PREDICTION
The anomaly detection model of spacecraft telemetry data
based on TCN time series prediction is focused on in this
section. A framework is given in Fig. 5, and there are two main
units: the prediction unit and the anomaly detection unit. The
main TCN algorithm is described in Section II. Before training
and testing, the datasets need to be normalized. The proposed
anomaly detection method is illustrated as follows.

```
Telemetry
Datasets
```
```
Data
Preprocessing
```
```
Train datasets Train TCN/LSTM
```
```
Test datasets
```
```
Training
Model
```
```
Predicted Data
ŷt
```
```
Test TCN/LSTM
```
```
Original Data yt
```
```
Error e=ŷt -yt Threshold caculation Anomaly interval
```
```
Prediction Unit
```
```
Anomaly detection Unit
```
```
Performance
evaluation
```
Fig. 5. The framework of anomaly detection

The predicted telemetry values are gained from the
training and testing phase. The statistics method is commonly
used to detect anomalies. Because the original data are not
stable, it is not suitable for static threshold method. Thus, the
residual between actual data and predicted data is applicable
to the mentioned method due to its stability. At first, calculate
the difference between the actual telemetry values and the
predicted telemetry values to obtain a set of error sequence

```
ee () tt  yt y. Then, the error sequence is compared with
```
upper threshold and lower threshold which are constant. The
mean and variance of the residual from training data are
essential statistics of calculating thresholds. Taking an

## example of

```
t
e^ which is the residual sequence between
```
original data and predicted data in training phase, its

## corresponding threshold *

```
n
```
##### H th is defined as following:

```
* up =
```
##### HP G th eze

(^) 

## G e

## G (4)

## * =

```
down
```
##### HP G th eze

(^)  GG _e_ (^) (5)

### where P e^ is the mean of e^. G e^ is the standard

##### deviation of e^. z is the coefficient to trade off mean and

```
standard deviation. Generally, z is the confidence level under
the 99.7% confidence interval which obeys normal
```
## distribution. If the error exceeds the upper threshold *

```
up
```
##### H th

## or below the down threshold *

```
down
```
##### H th , the data at this

```
position is considered as an abnormal value.
```
```
IV. EXPERIMENTS AND RESULTS ANALYSIS
```
```
A. Data Description and Evaluation Metrics
A set of real telemetry datasets of solid state power
amplifier which are transmitted from two different satellites
are used for experiments to evaluate the performance of TCN.
The variables of experimented data from the first satellite are
voltages and currents, and the data vary in a periodic trend.
There are enough samples to show the periodicity in the
training datasets. The variables of experimented data from the
other satellite are currents and temperatures, and the data are
in a long-term trending variation. Also, there are enough
samples to show the rising or descending trend in the training
datasets. In addition, two common types of anomalies are
injected into the real data respectively to simulate anomalies:
point anomalies and contextual anomalies. Point anomalies
are individual data points which are different from remaining
normal data points among whole spacecraft data. The
abnormal data where the associated environment between
upper and lower of telemetry data is time are contextual
anomalies. Figure 6 gives an intuitive visualization of points
and contextual anomalies.
```
```
Metrics are important factors to evaluate the performance
of models. Accuracy ( ACC ), True Positive Ratio ( TPR ) and
False Positive Ratio ( FPR ) [14] are the common metrics used
for the anomaly detection. The propose of TCN is to save
model inference time, so the time consume is also an
important indicator. Specifically, time consume is the running
time to predict the test data.
```
```
ACC , TPR and FPR can be expressed as follows:
```
```
TP TN
ACC
TN FN TP FP
```
###### 

###### 

###### (6)

###### TP

###### TPR

###### FN TP

###### 

###### (7)

###### FP

###### FPR

###### TN FP

###### 

###### (8)

##### where TP is the number of detected anomaly values in

##### abnormal samples. FP represents the normal data which is

```
detected as abnormal data. TP FN  is the number of all
actual abnormal data. FP TN  is the actual normal samples.
TN FN TP FP is the total number of all samples for
verification.
```
```
B.Experimental Results and Analysis
Due to the fact that LSTM is a widely used algorithm in
time series anomaly detection with excellent performance, a
set of comparative experiments is implemented based on the
```

LSTM algorithm. Reference [15] introduced its theories
detailly. In the experiment, considering different types of
telemetry data for universality of results, firstly, take
randomly three sets of telemetry data with short previous
timestep. Specifically, the parameter timestep is set to 20, and
take the average ( _AVG_ ) of metrics of _ACC_ , _TPR_ , _FPR_ and
model time consume. Take the battery voltage as an example,
its visual results of TCN and LSTM are shown as Fig. 6 and
Fig. 7 separately. The red dotted line represents predicted data,
and the blue solid line represents actual data. These two
figures state the ability of TCN to predict the periodicity of
telemetry data is as good as LSTM. Also, the difference of
predicted data and actual data based on TCN is obvious in
terms of anomalies, contextual anomalies in particular.

In order to obtain anomalies directly, the residuals of
predicted data and actual data are calculated. Then, the upper
and lower thresholds are obtained with the formular (5)
through the residual statistics. The residual curve of predicted
and actual values is shown in Fig. 8. The red line represents
the curve of residuals and the black line represents the upper
threshold and down threshold. Obviously, TCN obtains
prominent anomalies. However, it is difficult to observe the
positions and the specific number of detected anomalies from
the pictorial results. The metrics of telemetry data with short
timestep based on TCN and LSTM are shown on Table I,
which proves the performance of models convincingly. In
addition, metrics can quantify the points which exceed the
upper and lower boundaries in Fig.8 to determine actual
anomalies and false detection reflected from _TPR_ and _FPR_ of
voltage1 in Table I. _TPR_ of voltage1 based on TCN is higher
than that based on LSTM, and _FPR_ of voltage1 based on TCN
is lower than that based on LSTM, which demonstrates all
actual anomalies and fewer false detected points break the
boundaries based on TCN.

```
Point anomaly
```
```
Contextual anomaly
```
Fig. 6. Anomaly detection of voltage 1 based on TCN

Fig. 7. Anomaly detection of voltage 1 based on LSTM

```
Fig. 8. Residual of voltage 1 based on TCN and LSTM respectively
```
```
It can be seen from the Table I, TCN has an excellent
performance like LSTM in terms of ACC , TPR and FPR.
These three metrics indicate that the detection performance is
in high accuracy, high detection rate and low false detection.
In addition, both methods achieve a good performance in
model inference time, neither of which exceeds 1s. However,
the parallel ability of TCN is not reflected from the above
experiments. There is no verification on high operating
efficiency of TCN. The reason of this occurrence is that the
amount of previous data, i.e., timestep, is not as enough to
demonstrate the parallel feature. In order to explore the
parallel function and the learning ability of TCN, two sets of
experiments with large amount of data are conducted to ensure
sufficient training in the case of long timestep. The timestep is
set to 200 in this experiment. This set of data is current
telemetry data of solar panels with a descent trend. Figure 9
and 10 show the predicted data and actual test data as well as
residuals based on LSTM and TCN respectively. By checking
Fig.9, both methods have a good regression with a descend
trend. In Fig.10, it is obvious that both methods detect two
parts of anomalies which exceed the upper and lower
boundaries. The metrics of large telemetry data is given in
Table II.
```
```
TABLE I. METRICS OF ANOMALY DETECTION WITH SHORT
TIMESTEP
```
```
Model Metrics Voltage1 Voltage2 Telemetry data Current1 AVG
```
```
LSTM
```
```
ACC 9 4.86% 9 4.86% 9 7.64% 9 5.79%
TPR 9 8.18% 9 5.45% 1 00% 9 7.88%
FPR 5.76% 5 .18% 2 .52% 4 .49%
Op.Time 0 .098s 0.102s 0 .283s 0.161s
```
```
TCN
```
```
ACC 96% 9 3.4% 1 00% 9 6%
TPR 1 00% 9 5.45% 1 00% 9 8%
FPR 3 .73% 6 .7% 0 3.48%
Op.Time 0 .198s 0 .125s 0 .398s 0 .240s
```

Fig. 9. Anomaly detection of current based on TCN and LSTM respectively

Fig. 10. Residual of current based on TCN and LSTM respectively

```
TABLE II. METRICS OF ANOMALY DETECTION WITH LONG TIMESTEP
```
```
Model Metrics Current2 Telemetry data Temperature AVG
```
```
LSTM
```
```
ACC 99.42% 9 5.56% 9 7.49%
TPR 9 9.46% 9 9.73% 9 9.60%
FPR 0 .58% 4 .80% 2 .69%
Op.Time 8 .198s 1 .401s 4 .800s
```
```
TCN
```
```
ACC 99.07% 93.82% 96.45%
TPR 97.30% 96.43% 9 6.865%
FPR 0.71% 6.40% 3 .56%
Op.Time 0 .383s 0 .188s 0.571s
```
Table II indicates that the average of _ACC_ and _TPR_ based
on both models exceed 95% and the average of _FPR_ is below
5%, which remains good performance. The discrepancy of
model inference time between TCN and LSTM is significant.
TCN still keeps excellent property in model inference time,
but LSTM spends much more time with long previous
timestep than itself in the experiment with short timestep. It
can be proved that TCN is more significant in saving model
processing time and achieve a good detection performance.

V. CONCLUSION AND FUTURE WORK
This work achieves anomaly detection analysis of
spacecraft telemetry data transmitted from the on-orbit
satellite and evaluates the performance of TCN by being
compared with LSTM. Both models perform satisfied
detection performance reflected on _ACC_ , _TPR_ and _FPR_.

```
Moreover, TCN consumes less model inference time no
matter with short timestep or long timestep due to its parallel
ability. Generally, TCN can be applied to the anomaly
detection of spacecraft telemetry data with good detection
performance even high operating efficiency. Actually, the
proposed method is currently attempted to Beidou satellites
developed by Innovation Academy for Microsatellites of CAS.
More detailed test results will be reported in the next couple
of months to indicates its potential applicable capabilities.
```
```
Actual engineering applications generally need online
detection method due to the telemetry data of satellite is
transmitted as a real-time dataflow. The high accuracy and
operating efficiency of TCN are suitable for online detection.
In further, the proposed method will be improved for real
satellite online anomaly detection, i.e., multiple variables of
telemetry data, and complex anomaly modes.
## BibTeX (optional)

```bibtex
@inproceedings{wang2021anomaly,
  title={Anomaly Detection of Spacecraft Telemetry Data Using Temporal Convolution Network},
  author={Wang, Yuan and Wu, Yan and Yang, Qiong and Zhang, Jun},
  booktitle={2021 IEEE International Instrumentation and Measurement Technology Conference (I2MTC)},
  pages={1--6},
  year={2021},
  organization={IEEE},
  doi={10.1109/I2MTC50364.2021.9459840}
}
```
