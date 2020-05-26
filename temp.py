import torch
import time
print(torch.cuda.is_available())

for i in range(3):
    time.sleep(1)
    print("diri"+str(i))
   
