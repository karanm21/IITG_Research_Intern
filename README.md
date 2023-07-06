# IITG_Research_Intern
**Enabling Low Power Cardiac Arrhythmia Classification on Wearable
Devices**
--------
**Cardiovascular diseases** (CVD) are one of the primary causes of mortality worldwide, causing millions
of casualties every year. Among all the chronic CVDs, **cardiac arrhythmia** is the most recurrent and is
responsible for the maximum number of fatalities due to cardiac arrests. Thus, the development of
wearable devices for the timely detection of CA would enable on-time treatment to save millions of
lives.

A **Deep Neural Network** (DNN) based cardiac arrhythmia (CA) classifier is suggested in this research,
and it can categorize ECG beats into normal and various types of arrhythmia beats. A time domain
ECG signal's optimized fixed length beat is taken out and used as an input by the suggested classifier.
This predetermined input beat size facilitates the optimization of our suggested architecture and
eliminates the need to manually extract ECG features. 

The classifier that we are building does not make much use of complex algorithms for CA
classification. Thus, the low power realization of the proposed system makes it suitable for **wearable
healthcare device** applications.

## Data Description
The **MIT-BIH Arrhythmia Database** contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979. Twenty-three recordings were chosen at random from a set of 4000 24-hour ambulatory ECG recordings collected from a mixed population of inpatients (about 60%) and outpatients (about 40%) at Boston's Beth Israel Hospital; the remaining 25 recordings were selected from the same set to include less common but clinically significant arrhythmias that would not be well-represented in a small random sample.

The recordings were digitized at **360 samples per second per channel** with 11-bit resolution over a 10 mV range.

