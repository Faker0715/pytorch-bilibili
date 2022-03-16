from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")

#四通道在open后加入.convert('RGB')转3通道
img = Image.open("dataset/val/ants/8124241_36b290d372.jpg")
print(img)

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)


print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm)
# 2*x - 1
print(img_norm[0][0][0])
writer.add_image("Normailze",img_norm,2)

print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL -> resize -> PIL -> tensor
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize)
print(img_resize)

# Compose --resize - 2
trans_resize_2 = transforms.Resize((512,512))
# PIL -> tensor 不能调换顺序 后面的输入是前面的输出
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize_2",img_resize_2,1)

# RandomCrop
trans_random = transforms.RandomCrop(100)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])

for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)


writer.close()
