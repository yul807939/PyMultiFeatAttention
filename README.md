

# Accurate Identification of Mastigophora Species via Multimodal Deep Learning: An Integrated Classification Model Combining Morphological and Molecular Data


# Abstract
Traditional classification of flagellates relies on morphological traits or single molecular markers, which suffer from subjectivity and limited data sources. This study proposes a multimodal deep learning model, RseMFA-50, that integrates microscopic images and SSU rRNA gene sequences of flagellates. The dual-branch architecture (DNA sequence and image branches) extracts local and global features, while the Multi-Feature Attention (MFA) mechanism dynamically fuses heterogeneous data. Experiments were conducted on a curated dataset comprising 184 SSU rRNA sequences and 321 standardized microscopic images, evaluated via 10-fold cross-validation. The results demonstrate that RseMFA-50 achieves an accuracy of 87.0% in classifying flagellates at a batch size of eight, which is significantly higher than the accuracies achieved by EfficientNet (81.4%), ResNet50 (84.1%), and MMNet (85.6%).  The dual-channel global pooling mechanism (GAP and GMP fusion) balances the variance-bias trade-off through complementary strategies of spatial statistical smoothing and local salient feature detection, enhancing robustness to data scale expansion. This study establishes a methodological foundation for modeling multimodal biological data in complex systems, advancing deep learning applications in integrative taxonomy.


# Overview
![Image text](https://github.com/yul807939/PyMultiFeatAttention/tree/main/pic/overview.png)
