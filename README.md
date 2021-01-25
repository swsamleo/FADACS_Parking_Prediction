# FADACS: A Few-shot Adversarial Domain Adaptation Architecture for Context-Aware Parking Availability Sensing


## Introduction
Parking availability sensing plays a vital role in urban planning and city management [24,37]. According to a recent study, drivers spend more than 100,000 hours per year in looking for parking their cars [27]. Moreover, seeking for available parking can leadto severe traffic congestion and air pollution [9]. Hence, effective parking availability sensing can help drivers find a vacant parking spot. This also helps government to take appropriate measure byunderstanding the utilisation of parking facilities and provide moreon-street parking lot in the areas with high parking demand

## ADACS ARCHITECTURE
The traditional method for transfer learning is fine-tuning, which first loads a pre-trained parameter from other tasks and then re-train them on the new domain/task. However, one issue that needs to be faced in real-world usage is that most of the task only hasfew or no historical data at all. According to [4], Tzenget al.[30]propose a general architecture for adversarial domain adaptionnamed Adversarial Discriminative Domain Adaption (ADDA). This new framework used in ADDA combines a discriminative model,untie weight sharing and GAN loss together, which shows a promising performance on unsupervised transfer learning. Compare to other domain adaptation methods, ADDA introduces the adversar-ial mechanism which trains an encoder to translate the featuresfrom the target domain to the latent space shared by both thesource and target domain. Meanwhile, a discriminator is trainedsimultaneously to distinguish the origin of each latent code.In this paper, we adopt the original ADDA framework, which isinitially used for image classification task and modify it to makesit applicable for our time-series prediction problem. We useXsandXtto donate source and target domain features.Ysdenotesoccupancy rate of parking lots from the source domain.Ms(Xs)donates source mapping/encoder andMt(Xt)is about target map-ping. The regression model is represented asFwhileDstands forthe discriminator. The architecture we use in this paper is shownin Figure 3, and it comprises the three following stages.The first part is the pre-training step to learn a source encoderMs(Xs)and a regression model based on the source domain data.Similar to an auto-encoder structure, the encoder here learns amapping the source domain to a latent space. On the other hand,the regressor learns to decode features from this latent space andmake a prediction on top of that. We use ConvLSTM (ConvolutionalLong-Short Term Memory) proposed in [33] as the encoder, whichshows a good performance on spatio-temporal data. Extending ona common LSTM unit, matrix multiplication is replaced by convo-lution operation at each gate in the LSTM cell. The key equationsof ConvLSTM are shown in equation 1 below, where ’∗’ denotesthe convolution operator and ’◦’ denotes the Hadamard product:
<p align="center"><img width="100%" height="100%" src="images/f3.png"></p>

Next step is an adversarial adaptation, which is to learn a targetencoderMt(Xt)so that the discriminatorDcannot distinguish theorigin of that sample. By fixing source encoder parameter, the adver-sarial loss is used to minimise the distance of the mapping betweensource and target domain:Ms(Xs)andMt(Xt)and maximise thediscriminator loss.

<p align="center"><img width="70%" height="100%" src="images/f2.png"></p>

In the final stage, we assemble the learned target encoderMt(Xt)and regression modelFtogether, and use data from the target domain to test its performance. The regressor should the ability to generate quality prediction since the latent features from the target domain is overlapping with the ones from the source domain after the previous adaptation stage.


## Results

In the first experiment, we compare a couple of existing approachesto predict the parking occupancy. We select four classic approachhere: HA, MLP, LSTM and ConvLSTM. HA is a basic statisticalmethod to estimate the parking occupancy based on the historicaldata by averaging them. The strength of this method is that HA cancatch the periodical pattern of parking occupancy. However, it doesnot consider spatial dependency, temporal dependency and hiddentrends in the data. Compared to HA, MLP can automatically explorethe trends of the parking occupancy even though it also does notconsider the spatio-temporal dependency. LSTM can predict theparking occupancy by leveraging the temporal dependency of thehistorical data which is the essential to time-series data prediction.However, as we mentioned in the introduction, the parking sens-ing not only relies on the temporal dependency but also relevantto the spatial dependency. ConvLSTM can integrate spatial andtemporal features into one simple end-to-end model and Table 5also validates our assumption. In Table 5, ConvLSTM outperformsother classic parking prediction approach for all prediction hori-zons. LSTM outperforms the second since it consider the temporaldependency but not spatial dependency. MLP performs better thanHA but lose the match to LSTM and ConvLSTM. This result sug-gests us that both spatial and temporal dependency play a role inthe parking occupancy prediction, and the temporal dependencyseems more important since the gap between the LSTM and MLPis much smaller than MLP and other approaches.
<p align="center"><img width="100%" height="100%" src="images/f4.png"></p>


The first experiment shows that ConvLSTM perform the best inparking sensing. Then, we conduct a few-shot transfer learning testto validate the effectiveness of our proposed transfer learning modelwith a few training samples from the target domain. most machinelearning techniques require thousands of examples to achieve goodperformance in parking prediction. The goal of few-shot learningis to achieve acceptable accuracy in parking sensing with a fewtraining examples in target domain. We compare our model to fourclassic approaches used in spatio-temporal transfer learning area:LSTM with parameter transfer, ConvLSTM with parameter transfer,ADDA with MLP and our propose architecture. The first and secondmodel are based on parameter transfer framework, which transferthe parameters trained in the source domain to the target domain.ADDA with MLP and our proposed architecture are GAN-basedtransfer learning framework. Table 6 shows that our approachperform the best. The ConvLSTM with parameter transfer performbetter than LSTM with parameter transfer, and the ADDA with MLPperform the worst. This result validates our claim that both spatialand temporal dependency are significantly important in parkingoccupancy prediction, and adversarial learning is a good at learningthe shared feature spaces. Additionally, it again validates that theimportance of each component should be temporal dependency,spatial dependency and domain adaption.In summary, we have conducted two experiments with Mel-bourne CBD parking data, Rye parking data and multiple contex-tual features. The experimental results show that our approachwhich integrates spatial information, temporal information anddomain adaption outperform other baselines. It also shows the im-portance of each component in predict parking occupancy in targetdomain by leveraging source domain historical data and contextualinformation.

## Prerequisites
Our code is based on Python3 (>= 3.8). Here is the dependencies to run the code. The major libraries are listed as follows, more detail please check the requirement.txt
* TensorFlow (>= 2.1.0)
* Torch (>=1.60)
* NumPy (>= 1.17.3)
* SciPy (>= 1.4.1)
* Pandas (>= 1.0.1)

## Dataset
The two cities that are being investigated in this research are theCity of Melbourne and the town of Rye. Both are in the state ofVictoria, Australia. Melbourne is the capital city of Victoria. Themunicipality of Melbourne, with an estimated of 178,955 residents[12], has nearly 1 million people on average per day, visiting themunicipality for work, education, and travel or tourism. On theother hand, Rye is a little coastal town, part of the MorningtonPeninsula Shire municipality. Rye has a population of approximately8,416 in the 2016 census and is located about 100km from the City ofMelbourne. The Mornington Peninsula Shire hosts about 7.5 millionvisitors per year [26], and about 50% of those would visit Rye asone of their destinations, requiring parking spot, as driving is themain mode of transport to get into these coastal areas. Therefore,the major datasets in this paper include parking sensor data, Pointsof Interests (POI) data, weather data, and geographical data.All the datasets used in this system come from the followingplatforms: the City of Melbourne Open Data [13], Time and DateAS [29], Google Map API, and a proprietary Mornington PeninsulaShire data platform

### Pre-processed Data
[Download From Google Drive](https://drive.google.com/drive/folders/1ARLiwHIezdkHiT7tTzOS84vvbVJkLvn1?usp=sharing)


## Experiment Details

### Run FADACS Experiment
1. Unzip the dataset.7z file in experiments dir first
    ```
        ├── experiments
        │   └── FADACS
        │       ├── exp.json
        │       ├── MelbCity
        │       │   ├── testIndex.npy
        │       │   ├── trainIndex.npy
        │       │   ├── x.npy
        │       │   └── y.npy
        │       └── Mornington
        │           ├── testIndex.npy
        │           ├── trainIndex.npy
        │           ├── x.npy
        │           └── y.npy
    ```
2. Than run it
    ```python
    python run_FADACS.py
    ```

Or conduct experiments as you wish

1. Download the Pre-processed Data and unzip it first.
2. modify run_experiment_example.py then run it.

    ```python
    python run_experiment_example.py
    ```

## Updates
**Jan. 22, 2021**:
* All dataset of Mornington(Rye) has been removed.

## Citation
Please refer to our paper. Wei Shao*, Sichen Zhao*, Zhen Zhang, Shiyu Wang , Saiedur Rahaman, Andy Song , Flora D Salim. FADACS: A Few-shot Adversarial Domain Adaptation Architecture for Context-Aware Parking Availability Sensing. In *2021 IEEE International Conference on Pervasive Computing and Communications (PerCom) - IEEE International Conference on Pervasive Computing and Communications 2021

    @INPROCEEDINGS{Shao2103:FADACS,
    AUTHOR="Wei Shao and Sichen Zhao and Zhen Zhang and Shiyu Wang and Mohammad
    {Saiedur Rahaman} and Andy Song and Flora D Salim",
    TITLE="{FADACS:} A Few-shot Adversarial Domain Adaptation Architecture for
    {Context-Aware} Parking Availability Sensing",
    BOOKTITLE="2021 IEEE International Conference on Pervasive Computing and
    Communications (PerCom) (PerCom 2021)",
    ADDRESS="Kassel, Germany",
    DAYS=21,
    MONTH=mar,
    YEAR=2021,
    ABSTRACT="The existing research on parking availability sensing mainly relies on
    extensive contextual and historical information. In practice, it is
    challenging to have such information available as it requires continuous
    collection of sensory signals. In this paper, we design an end-to-end
    transfer learning framework for parking availability sensing to predict the
    parking occupancy in areas where the parking data is insufficient to feed
    into data-hungry models. This framework overcomes two main challenges: 1)
    many real-world cases cannot provide enough data for most existing
    data-driven models. 2) it is difficult to merge sensor data and
    heterogeneous contextual information due to the differing urban fabric and
    spatial characteristics. Our work adopts a widely-used concept called
    adversarial domain adaptation to predict the parking occupancy in an area
    without abundant sensor data by leveraging data from other areas with
    similar features. In this paper, we utilise more than 35 million parking
    data records from sensors placed in two different cities, one is a city
    centre, and another one is a coastal tourist town. We also utilise
    heterogeneous spatio-temporal contextual information from external
    resources including weather and point of interests. We quantify the
    strength of our proposed framework in different cases and compare it to the
    existing data-driven approaches. The results show that the proposed
    framework outperforms existing methods and also provide a few valuable
    insights for parking availability prediction."
    }
