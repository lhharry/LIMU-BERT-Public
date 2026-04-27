import torch

sd = torch.load("D:\\01_Code\\LIMU-BERT-Public\\saved\\pretrain_base_camargo_20_120\\limu_bert_x.pt", map_location="cpu")
for k, v in sd.items():
    if "pos" in k.lower() or "embed" in k.lower():
        print(k, v.shape)