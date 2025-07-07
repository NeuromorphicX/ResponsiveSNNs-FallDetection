# 🧠 ResponsiveSNNs

**Improving Responsiveness of Fall Detection using Spiking Neural Networks**  
_Published in the 2025 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops)_

📄 [Paper Link](https://ieeexplore.ieee.org/abstract/document/11038662)

---

## 🔍 Abstract

Fall detection systems, essential for the safety of elderly individuals, have increasingly incorporated Deep Neural Networks (DNNs) for improved accuracy. However, real-time processing—especially on resource-constrained wearable devices—remains a challenge due to the computational demands of DNNs.

Recently, Spiking Neural Networks (SNNs) have shown promise for improving energy efficiency in fall detection. Yet, their potential to leverage **temporal dynamics** for faster responsiveness has not been fully explored.

This work proposes:

- **Quick Spike Encoding (QSE):** An encoding method that prioritizes critical inputs based on amplitude over time.
- **Linear Weighted Mean Squared Error Count (LW-MSEC) Loss Function:** A loss function that emphasizes early detection by penalizing early-stage errors more heavily.

Evaluations on two fall detection datasets show significant improvements:
- **91% accuracy in just 10 time steps** on SisFall (vs 25 steps in standard baselines)
- **Over 60% improvement in responsiveness**

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@INPROCEEDINGS{11038662,
  author={Sabbella, Hemanth and Mukherjee, Archit and Chuang, Tan Jeck and Yee Low, Hong and Ma, Dong and Misra, Archan},
  booktitle={2025 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops)}, 
  title={Improving Responsiveness of Fall Detection using Spiking Neural Networks}, 
  year={2025},
  pages={98--103},
  doi={10.1109/PerComWorkshops65533.2025.00048},
  keywords={Accuracy;Conferences;Spiking neural networks;Encoding;Real-time systems;Timing;Safety;Fall detection;Wearable devices;Older adults;Spiking Neural Networks;Responsiveness;Spike Encoding;Weighted Loss Function}
}
