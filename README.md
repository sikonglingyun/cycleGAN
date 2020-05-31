1. I changed normalization method from BatchNormarization2d to InstanceNorm2d.

2. ResNet helped deep neural network training

3. Discrimination loss function adopt MSE not BCE

4. But cycle-loss function is MAE not MSE

5. cycle-late : is the balance of cycle and GAN, the value is just "1", if the value of it is "10",the network trained equal Mapping