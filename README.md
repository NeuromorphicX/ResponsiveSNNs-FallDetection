# ResponsiveSNNs
This is the final code base for the paper "Improving Responsiveness of Fall Detection using Spiking Neural Networks" published in the  2025 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops). 


## Abstract

Fall detection systems, essential for the safety of elderly individuals, have increasingly incorporated Deep Neural Networks (DNNs) for improved accuracy. However, real-time processing, especially on resource-constrained wearable devices, remains a major challenge due to the computational intensity of DNNs. Recently, Spiking Neural Networks (SNNs) have shown promise in improving energy efficiency in fall detection. Despite these advancements, the potential of SNNs to leverage their temporal dynamics—particularly, the ability to make sequential decisions based on the timing of input spikes—has not been fully explored for enhancing detection responsiveness. This paper proposes a novel approach to exploit the temporal nature of SNNs for faster fall detection. Specifically, we introduce a novel encoding technique, Quick Spike Encoding (QSE), which prioritizes critical inputs based on signal amplitude in temporal dimension, and a Linear Weighted Mean Squared Error Count (LW-MSEC) Loss Function, which emphasizes early detection by penalizing errors more heavily at the beginning of the detection process. Evaluations on two fall detection datasets demonstrate that our method significantly enhances the responsiveness of SNNs (>60%), achieving 91% accuracy in just 10 time steps compared to 25 time steps with traditional approaches, on the SisFall dataset.

Paper link: https://ieeexplore.ieee.org/abstract/document/11038662 

## Citation 

if you use this code in your reaseach please cite: 


```bibtex

@INPROCEEDINGS{11038662,
  author={Sabbella, Hemanth and Mukherjee, Archit and Chuang, Tan Jeck and Yee Low, Hong and Ma, Dong and Misra, Archan},
  booktitle={2025 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops)}, 
  title={Improving Responsiveness of Fall Detection using Spiking Neural Networks}, 
  year={2025},
  volume={},
  number={},
  pages={98-103},
  keywords={Accuracy;Conferences;Spiking neural networks;Encoding;Real-time systems;Timing;Safety;Fall detection;Wearable devices;Older adults;Spiking Neural Networks;Responsiveness;Spike Encoding;Weighted Loss Function},
  doi={10.1109/PerComWorkshops65533.2025.00048}}
