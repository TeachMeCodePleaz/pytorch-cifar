import torch

# 指定.pth文件的路径
checkpoint_path = 'C:/Users/sodiu/Desktop/pytorch-cifar/checkpoint/ckpt.pth'

# 加载.pth文件
checkpoint = torch.load(checkpoint_path)

# 打印出.pth文件的内容
print("Model State Dict Keys:", checkpoint.keys())

# 打印模型权重的键名
if 'net' in checkpoint:
    print("Model State Dict:")
    for key in checkpoint['net'].keys():
        print(key)

# 打印其他信息
if 'acc' in checkpoint:
    print("Accuracy:", checkpoint['acc'])
if 'epoch' in checkpoint:
    print("Epoch:", checkpoint['epoch'])