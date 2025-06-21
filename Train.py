import torch
import torch.nn as nn
from Dataset import Dataset_y
from torch.utils.data import DataLoader
from model import ResNet_F_D,BasicBlock
dev = 6
epoch_num =  100

result = []





if __name__ == "__main__":

    
    total_acc = 0
    for test_num in range(10):
        model = []
        model = ResNet_F_D(BasicBlock, [3, 4, 6, 3], 123)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
        model.to(dev)
        optim = torch.optim.Adam(model.parameters(), lr=0.001 , )

        
        Traindata = Dataset_y( "./Data/tu/" , test_num, True)
        Testdata = Dataset_y( "./Data/tu/" , test_num,  False)
        Train_loader = DataLoader(Traindata, 32,True,  num_workers=0)
        Test_loader = DataLoader(Testdata, 32,True,  num_workers=0)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(epoch_num):

            num = 0
            for pic ,seq, label in Train_loader:
                num += 1
                pic = pic.to(6)
                
                seq = seq.to(6 , dtype = torch.float32)
                seq = seq.unsqueeze(1)

                if epoch % 50 == 0 :
                    for param_group in optim.param_groups:
                        param_group['lr'] = 0.0001

                pred = model(pic ,seq )
                l = label.long() - 1
                one = nn.functional.one_hot(l , num_classes=123)
                one = one.to(device=dev,dtype= torch.float32)
                loss = loss_func(pred , one)
           

                loss.backward()
                optim.step()
                optim.zero_grad()

            with torch.no_grad():
                total_acc = 0

                for  pic ,seq ,label in Test_loader:
                    pic = pic.to(dev)
                    seq = seq.to(dev , dtype = torch.float32)
                    seq = seq.unsqueeze(1)
                    pred = model(pic,seq )
                    pred = torch.argmax(pred , dim =1) 
                    label = label.to(dev) - 1
                    acc = (pred == label).sum()
                    total_acc = total_acc + acc
                print("PIC_&_DNA_new %3d : "%(epoch + 1) , total_acc / len(Testdata) )

        result.append(total_acc / len(Testdata))    

    print("Only_DNA_new_FINAL RESULT " , result)












