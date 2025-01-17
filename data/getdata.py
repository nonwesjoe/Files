import os
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms,datasets
from PIL import Image
from tqdm import tqdm
import torch
import math

def one_hot_collate(batch,num_classes):
    images, labels = zip(*batch)  # 分离图像和标签
    images = torch.stack(images)  # 将图像堆叠为一个张量
    labels = torch.tensor(labels)  # 将标签转换为张量
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()  # 转为 one-hot
    return images, one_hot_labels

def example(loader):
    x,y= next(iter(loader))
    print(f'\n====detail of  your dataset====\n')
    print(f'shape of x : {x.shape}')
    print(f'shape of y : {y.shape}')
    print(f'\n=====take a look of your dataset====\n')
    print(f' an example x : {x[0]}')
    print(f'-----------------------------')
    print(f' an exmaple y : {y[0]}')
    print(f'\n====good luck to your DL journey====\n')

class Get_data():
    print('wsz data geter is ready')
    def __init__(self,batch_size=32,size=224,datapath=r"./"):
        self.datapath=datapath
        self.era5path=os.path.join(self.datapath,"era-shenzhen.h5")
        self.fishpath=os.path.join(self.datapath,"8-fish")
        self.flowerpath=os.path.join(self.datapath,"14-flowers")
        self.fruitpath=os.path.join(self.datapath,"17-fruits")
        self.animalpath=os.path.join(self.datapath,"9-animals")
        self.segmentpath=os.path.join(self.datapath,"segment-diatom")
        self.size=size
        self.batch_size=batch_size
        self.transform=transforms.Compose([
            transforms.Resize((self.size, self.size)),  # 调整图像大小
            transforms.ToTensor(),         # 转换为 PyTorch Tensor
            transforms.Normalize([0.5], [0.5])  # 归一化 (mean=0.5, std=0.5)
        ])


    def get_era5(self):
        with h5py.File(self.era5path, 'r') as f:
            data = f['data'][:]
        print(f'get data from path {self.era5path}')
        print(f"era5 data shape: {data.shape}")
        print(f'an example of data {data[0]}')
        return data
    
    def get_fish(self):
        dataset = datasets.ImageFolder(root=self.fishpath, transform=self.transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        num_classes = len(dataset.classes)
        trainload = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=lambda batch: one_hot_collate(batch,num_classes))
        valload = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,collate_fn=lambda batch: one_hot_collate(batch,num_classes))
        print(f'get data from path {self.fishpath}')
        print('fish data get')
        example(trainload)
        return trainload,valload

    def get_flower(self):
        trianpath = os.path.join(self.flowerpath, 'train')
        valpath = os.path.join(self.flowerpath, 'val')

        train = datasets.ImageFolder(root=trianpath, transform=self.transform)
        val = datasets.ImageFolder(root=valpath, transform=self.transform)

        trainload = DataLoader(train, batch_size=self.batch_size, shuffle=True,collate_fn=lambda batch: one_hot_collate(batch, len(train.classes)))
        valload = DataLoader(val, batch_size=self.batch_size, shuffle=False,collate_fn=lambda batch: one_hot_collate(batch, len(val.classes)))
        print(f'get data from path {self.flowerpath}')
        print('flower data get')
        example(trainload)
        return trainload,valload
    
    def get_fruits(self):
        dataset = datasets.ImageFolder(root=self.fruitpath, transform=self.transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        trainload = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=lambda batch: one_hot_collate(batch, len(dataset.classes)))
        valload = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,collate_fn=lambda batch: one_hot_collate(batch, len(dataset.classes)))
        print(f'get data from path {self.fruitpath}')
        print('fruits data get')
        example(trainload) 
        return trainload,valload
    
    def get_segment_data(self):        
        class CustomDataset(Dataset):
            def __init__(self, x_tensor, y_tensor):
                self.x_tensor = x_tensor
                self.y_tensor = y_tensor

            def __len__(self):
                return len(self.x_tensor)

            def __getitem__(self, idx):
                return self.x_tensor[idx], self.y_tensor[idx]
            
        transform= transforms.Compose([ transforms.Resize((self.size, self.size)),  transforms.ToTensor()])
        xpath=os.path.join(self.segmentpath,"origin")
        ypath=os.path.join(self.segmentpath,"mask")
        x_list = []
        y_list = []
        # 遍历路径加载图片
        trpath = os.listdir(xpath)
        tepath = os.listdir(ypath)

        for tr, te in tqdm(zip(trpath, tepath)):
            trimagepath = os.path.join(xpath, tr)
            teimagepath = os.path.join(ypath, te)

            # 加载输入图片
            p = Image.open(trimagepath).convert('RGB')  # 确保为 RGB 模式
            p = transform(p)
            x_list.append(p)

            # 加载目标图片
            v = Image.open(teimagepath).convert('L')  # 灰度模式
            v = transform(v)
            y_list.append(v)

        # 转换为 PyTorch 张量
        x_tensor = torch.stack(x_list)  # 输入图片的张量
        y_tensor = torch.stack(y_list)  # 标签图片的张量

        # 创建数据集
        dataset = CustomDataset(x_tensor, y_tensor)

        validation_split = 0.2
        dataset_size = len(dataset)
        val_size = math.ceil(dataset_size * validation_split)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        trainload = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valload = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        print(f'get data from path {self.segmentpath}')
        print('segment data get')
        print('segment diatom data get')
        example(trainload)
        return trainload,valload

    def get_animal(self):
        trainpath=os.path.join(self.animalpath,"train")
        valpath=os.path.join(self.animalpath,"valid")
        traincsvpath=os.path.join(self.animalpath,"train_classes.csv")
        valcsvpath=os.path.join(self.animalpath,"val_classes.csv")

        class CustomImageDataset(Dataset):
            def __init__(self, csv_file, root_dir, transform=None):
                self.data = pd.read_csv(csv_file)
                self.root_dir = root_dir
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()

                img_name = self.data.iloc[idx, 0] 
                labels = self.data.iloc[idx, 1:].values.astype('float32')
                img_path = f"{self.root_dir}/{img_name}"
                
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
                return image, labels

        trainset = CustomImageDataset(csv_file=traincsvpath, root_dir=trainpath, transform=self.transform)
        valset = CustomImageDataset(csv_file=valcsvpath, root_dir=valpath, transform=self.transform)
        # 创建数据加载器
        trainload = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        valload = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        print(f'get data from path {self.animalpath}')
        print('animal data get')
        example(trainload)
        return trainload,valload


