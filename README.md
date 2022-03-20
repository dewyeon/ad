# Multiscale Anomaly detection

- supports both MVTec-AD and CIFAR-10 dataset
- MVTec-AD detection, localization
- CIFAR-10 detection (One-Class Classification / OCC)

### MVTec-AD
`python main.py --dataset mvtec --phase test --class_name bottle`

### CIFAR-10
`python main.py --dataset cifar10 --phase test --label 1 --load_size 32 --input_size 32`
