import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm




# https://velog.io/@pre_f_86/UNET-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0


class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        def cbr(in_channels, out_channels, kernel_size = 3, stride = 1):
            layers = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            return layers
        
        self.enc1_1 = cbr(3,64)
        self.enc1_2 = cbr(64,64)
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.enc2_1 = cbr(64,128)
        self.enc2_2 = cbr(128,128)
        self.pool2 = nn.MaxPool2d(2,2)

        self.enc3_1 = cbr(128,256)
        self.enc3_2 = cbr(256,256)
        self.pool3 = nn.MaxPool2d(2,2)

        self.enc4_1 = cbr(256,512)
        self.enc4_2 = cbr(512,512)
        self.pool4 = nn.MaxPool2d(2,2)

        self.enc5_1 = cbr(512,1024)
        self.enc5_2 = cbr(1024,1024)

        self.unpool4 = nn.ConvTranspose2d(1024,512,2,2) ### 해당 부분 수정?
        self.dec4_2 = cbr(1024,512)
        self.dec4_1 = cbr(512,512)

        self.unpool3 = nn.ConvTranspose2d(512,256,2,2)
        self.dec3_2 = cbr(512,256)
        self.dec3_1 = cbr(256,256)

        self.unpool2 = nn.ConvTranspose2d(256,128,2,2)
        self.dec2_2 = cbr(256,128)
        self.dec2_1 = cbr(128,128)

        self.unpool1 = nn.ConvTranspose2d(128,64,2,2)
        self.dec1_2 = cbr(128,64)
        self.dec1_1 = cbr(64,64)

        self.result = nn.Sequential(
            nn.Conv2d(64,1,3,1,1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        #print(x.shape)
        enc1_1 = self.enc1_1(x) 
        enc1_2 = self.enc1_2(enc1_1) 
        pool1 = self.pool1(enc1_2) 
        
        
        enc2_1 = self.enc2_1(pool1) 
        enc2_2 = self.enc2_2(enc2_1) 
        pool2 = self.pool2(enc2_2) 
        
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1) 
        pool3 = self.pool3(enc3_2)        
        
        enc4_1 = self.enc4_1(pool3) 
        enc4_2 = self.enc4_2(enc4_1) 
        pool4 = self.pool4(enc4_2) 
        
        enc5_1 = self.enc5_1(pool4) 
        enc5_2 = self.enc5_2(enc5_1) 
        
        unpool4 = self.unpool4(enc5_2)

        #print(unpool4.shape, enc4_2.shape)
        ### 홀수로 인한 사이즈 맞춤. # 32,32 -> 33,33 
        unpool4 = F.interpolate(unpool4, size=(enc4_2.size(2), enc4_2.size(3)), mode='bilinear', align_corners=True)
        ###
        #print(unpool4.shape, enc4_2.shape)
                
        dec4_2 = self.dec4_2(torch.cat((unpool4,enc4_2),1))
        dec4_1 = self.dec4_1(dec4_2) 
        
        unpool3 = self.unpool3(dec4_1) 
        dec3_2 = self.dec3_2(torch.cat((unpool3,enc3_2),1)) 
        dec3_1 = self.dec3_1(dec3_2) 
        
        unpool2 = self.unpool2(dec3_1) 
        dec2_2 = self.dec2_2(torch.cat((unpool2,enc2_2),1)) 
        dec2_1 = self.dec2_1(dec2_2) 
        
        unpool1 = self.unpool1(dec2_1) 
        dec1_2 = self.dec1_2(torch.cat((unpool1,enc1_2),1))
        dec1_1 = self.dec1_1(dec1_2) 

        out = self.result(dec1_1)
        return out 




class Hybrid(nn.Module):
    def __init__(self):
        super(Hybrid,self).__init__()
        def cbr(in_channels, out_channels, kernel_size = 3, stride = 1):
            layers = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            return layers
        
        
        resnet = timm.create_model('resnet50', pretrained=True)
        #self.cnn_encoder = nn.Sequential(*list(self.cnn_encoder.children())[:-2]) # 2048,11,11

        # Split ResNet50 into separate blocks for skip connections
        self.resnet_layer0 = nn.Sequential(*list(resnet.children())[:4])  # initial conv + bn + relu + maxpool
        self.resnet_layer1 = resnet.layer1  # first block
        self.resnet_layer2 = resnet.layer2  # second block
        self.resnet_layer3 = resnet.layer3  # third block
        self.resnet_layer4 = resnet.layer4  # fourth block
        
        
        resnet_layer1_out_channels = list(resnet.layer1.children())[-1].bn3.num_features
        resnet_layer2_out_channels = list(resnet.layer2.children())[-1].bn3.num_features
        resnet_layer3_out_channels = list(resnet.layer3.children())[-1].bn3.num_features
        resnet_layer4_out_channels = list(resnet.layer4.children())[-1].bn3.num_features
        
        transformer_encoder = timm.create_model('pvt_v2_b0', pretrained=True)
        
        self.transformer_patch_emb = nn.Sequential( *list(transformer_encoder.children()) )[0]
        self.transformer_layer1 = nn.Sequential( *list(transformer_encoder.children()) )[1][0]
        self.transformer_layer2 = nn.Sequential( *list(transformer_encoder.children()) )[1][1]
        self.transformer_layer3 = nn.Sequential( *list(transformer_encoder.children()) )[1][2]
        self.transformer_layer4 = nn.Sequential( *list(transformer_encoder.children()) )[1][3] # 256,11,11
        
        transformer_layer1_out_channels = self.transformer_layer1.norm.normalized_shape[0]
        transformer_layer2_out_channels = self.transformer_layer2.norm.normalized_shape[0]
        transformer_layer3_out_channels = self.transformer_layer3.norm.normalized_shape[0]
        transformer_layer4_out_channels = self.transformer_layer4.norm.normalized_shape[0]
        
        self.cnn_down1 = nn.Conv2d(resnet_layer1_out_channels,transformer_layer1_out_channels,1,1)
        self.cnn_down2 = nn.Conv2d(resnet_layer2_out_channels,transformer_layer2_out_channels,1,1)
        self.cnn_down3 = nn.Conv2d(resnet_layer3_out_channels,transformer_layer3_out_channels,1,1)
        self.cnn_down4 = nn.Conv2d(resnet_layer4_out_channels,transformer_layer4_out_channels,1,1)
        
        del resnet; del transformer_encoder
        
        
        self.dec5_2 = cbr(transformer_layer4_out_channels*2,  512)
        self.dec5_1 = cbr(512,256)   
        
        self.dec4_2 = cbr(transformer_layer3_out_channels*2+256, 
                          transformer_layer3_out_channels*2+256)
        self.dec4_1 = cbr(transformer_layer3_out_channels*2+256,
                          256)

        self.dec3_2 = cbr(transformer_layer2_out_channels*2+256,
                          transformer_layer2_out_channels*2+256)
        self.dec3_1 = cbr(transformer_layer2_out_channels*2+256,
                          128)

        self.dec2_2 = cbr(transformer_layer1_out_channels*2+128,
                          transformer_layer1_out_channels*2+128)
        self.dec2_1 = cbr(transformer_layer1_out_channels*2+128,
                          64)
        
        self.dec1_2 = cbr(64,64)
        self.dec1_1 = cbr(64,32)

        self.result = nn.Sequential(
            nn.Conv2d(32,1,3,1,1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        #print(x.shape)
        
        #enc_cnn = self.cnn_encoder(x)
        res0 = self.resnet_layer0(x)
        res1 = self.resnet_layer1(res0) #256   
        res2 = self.resnet_layer2(res1) #512
        res3 = self.resnet_layer3(res2) #1024
        res4 = self.resnet_layer4(res3) #2048 

        res1 = self.cnn_down1(res1)
        res2 = self.cnn_down2(res2)
        res3 = self.cnn_down3(res3)
        res4 = self.cnn_down4(res4)
        
        #enc_transformer = self.transformer_encoder(x)
        #print(torch.cat((enc_cnn,enc_transformer),1).shape)
        
        trans0 = self.transformer_patch_emb(x) #352 16
        trans1 = self.transformer_layer1(trans0) #88 32
        trans2 = self.transformer_layer2(trans1) #44 64
        trans3 = self.transformer_layer3(trans2) #22 160
        trans4 = self.transformer_layer4(trans3) #11 256
        #print("tt")
        # print("trans0 : ", trans0.shape)
        # print("trans1 : ", trans1.shape)
        # print("trans2 : ", trans2.shape)
        # print("trans3 : ", trans3.shape)
        # print("trans4 : ", trans4.shape)
        
        
        #print('unpool5',res4.shape,trans4.shape)
        
        ## bridge
        
        # print(torch.cat((res4,trans4),1).shape)
        #unpool5 = self.unpool5(torch.cat((res4,trans4),1))
        dec5_2 = self.dec5_2(torch.cat((res4,trans4),1))
        #print(dec5_2.shape)
        dec5_1 = self.dec5_1(dec5_2)
        
        ###
        
        
        unpool4 = F.interpolate(dec5_1, size=(trans3.shape[2],trans3.shape[3]), mode='bilinear', align_corners=True)
        # print('unpool4',dec5_1.shape)
        # print(res3.shape,trans3.shape,unpool4.shape)
        dec4_2 = self.dec4_2(torch.cat((res3,trans3,unpool4),1))
        dec4_1 = self.dec4_1(dec4_2) 
        # print(dec4_1.shape)

        unpool3 = F.interpolate(dec4_1, size=(trans2.shape[2],trans2.shape[3]), mode='bilinear', align_corners=True)        
        dec3_2 = self.dec3_2(torch.cat((res2,trans2,unpool3),1)) 
        dec3_1 = self.dec3_1(dec3_2) 
        # print(dec3_1.shape)
        
        
        unpool2 = F.interpolate(dec3_1, size=(trans1.shape[2],trans1.shape[3]), mode='bilinear', align_corners=True)
        dec2_2 = self.dec2_2(torch.cat((res1,trans1,unpool2),1)) 
        dec2_1 = self.dec2_1(dec2_2) 
        # print(dec2_1.shape)
        
        unpool1 = F.interpolate(dec2_1, size=(x.shape[2],x.shape[3]), mode='bilinear', align_corners=True)
        dec1_2 = self.dec1_2(unpool1)
        dec1_1 = self.dec1_1(dec1_2) 
        # print(dec1_1.shape)
        out = self.result(dec1_1)
        
        return out 





def model_loader(config):
    if config.model == 'unet':
        model = UNet()
    if config.model == 'hybrid':
        model = Hybrid() # target
    else:
        model = UNet()
    return model


if __name__ == '__main__':
    pass
    #model = UNet()
    #model = Hybrid()
    #print(model)
    
    img = torch.randn(1,3,352,352)
    
    sample_model = UNet()
    sample_img = F.interpolate(img, scale_factor=0.75, mode='bilinear', align_corners=True)

    sample_output = sample_model(sample_img[0].unsqueeze(0))
    print(sample_output.shape)    
        