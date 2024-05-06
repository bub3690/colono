
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm



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
    
    
    
    
    class Hybrid(nn.Module):
        """
        Hybrid model using ResNet50 and PVTv2_b0
        Downsampling layer를 추가한 버전
        """
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
            
            self.cnn_down1 = nn.Conv2d(resnet_layer1_out_channels, transformer_layer1_out_channels,1,1)
            self.cnn_down2 = nn.Conv2d(resnet_layer2_out_channels, transformer_layer2_out_channels,1,1)
            self.cnn_down3 = nn.Conv2d(resnet_layer3_out_channels, transformer_layer3_out_channels,1,1)
            self.cnn_down4 = nn.Conv2d(resnet_layer4_out_channels, transformer_layer4_out_channels,1,1)
            
            del resnet; del transformer_encoder
            
            
            # Decoder concat downsample
            # Adaptive하게 2배씩 증가.
            target_decoder_size = 32
            self.decoder_down1 = nn.Conv2d(transformer_layer1_out_channels*2, target_decoder_size*2, 1, 1) # 64 -> 32
            self.decoder_down2 = nn.Conv2d(transformer_layer2_out_channels*2, target_decoder_size*2, 1, 1) # 128 -> 64
            self.decoder_down3 = nn.Conv2d(transformer_layer3_out_channels*2, target_decoder_size*2, 1, 1) # 320 -> 128
            #
            
            
            self.dec5_2 = cbr(transformer_layer4_out_channels*2,  512)
            self.dec5_1 = cbr(512,256)   
            
            self.dec4_2 = cbr(target_decoder_size*2+256, 
                            target_decoder_size*2+256)
            self.dec4_1 = cbr(target_decoder_size*2+256,
                            256)

            self.dec3_2 = cbr(target_decoder_size*2+256,
                            target_decoder_size*2+256)
            self.dec3_1 = cbr(target_decoder_size*2+256,
                            128)

            self.dec2_2 = cbr(target_decoder_size*2+128,
                            target_decoder_size*2+128)
            self.dec2_1 = cbr(target_decoder_size*2+128,
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
            
            

            ## bridge
            
            dec5_2 = self.dec5_2(torch.cat((res4,trans4),1))
            dec5_1 = self.dec5_1(dec5_2)
            
            ###
            
            
            unpool4 = F.interpolate(dec5_1, size=(trans3.shape[2],trans3.shape[3]), mode='bilinear', align_corners=True)
            
            dec4_skip = self.decoder_down3(torch.cat( (res3,trans3),1))
            dec4_2 = self.dec4_2(torch.cat( (dec4_skip,unpool4), 1) )        
            dec4_1 = self.dec4_1(dec4_2) 
            
            unpool3 = F.interpolate(dec4_1, size=(trans2.shape[2],trans2.shape[3]), mode='bilinear', align_corners=True)        
            dec3_skip = self.decoder_down2(torch.cat( (res2,trans2),1))
            dec3_2 = self.dec3_2(torch.cat((dec3_skip,unpool3),1)) 
            dec3_1 = self.dec3_1(dec3_2)  
            
            unpool2 = F.interpolate(dec3_1, size=(trans1.shape[2],trans1.shape[3]), mode='bilinear', align_corners=True)
            dec2_skip = self.decoder_down1(torch.cat( (res1,trans1),1))
            dec2_2 = self.dec2_2(torch.cat((dec2_skip, unpool2), 1)) 
            dec2_1 = self.dec2_1(dec2_2) 

            
            unpool1 = F.interpolate(dec2_1, size=(x.shape[2],x.shape[3]), mode='bilinear', align_corners=True)
            dec1_2 = self.dec1_2(unpool1)
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
        self.resnet_layer1 = resnet.layer1  # first blockc
        self.resnet_layer2 = resnet.layer2  # second blok
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
        
        
        self.cnn_down1 = nn.Conv2d(resnet_layer1_out_channels, transformer_layer1_out_channels,1,1)
        self.cnn_down2 = nn.Conv2d(resnet_layer2_out_channels, transformer_layer2_out_channels,1,1)
        self.cnn_down3 = nn.Conv2d(resnet_layer3_out_channels, transformer_layer3_out_channels,1,1)
        self.cnn_down4 = nn.Conv2d(resnet_layer4_out_channels, transformer_layer4_out_channels,1,1)
        
        del resnet; del transformer_encoder
        
        #### Multimodal Channel Attention
        
        self.mmtm1 = MMTM(resnet_layer1_out_channels, transformer_layer1_out_channels, 4)
        self.mmtm2 = MMTM(resnet_layer2_out_channels, transformer_layer2_out_channels, 4)
        self.mmtm3 = MMTM(resnet_layer3_out_channels, transformer_layer3_out_channels, 4)
        self.mmtm4 = MMTM(resnet_layer4_out_channels, transformer_layer4_out_channels, 4)        
        
        ####
        
        
        # Decoder concat downsample
        # Adaptive하게 2배씩 증가.
        target_decoder_size = 32
        self.decoder_down1 = nn.Conv2d(transformer_layer1_out_channels*2, target_decoder_size*2, 1, 1) # 64 -> 32
        self.decoder_down2 = nn.Conv2d(transformer_layer2_out_channels*2, target_decoder_size*2, 1, 1) # 128 -> 64
        self.decoder_down3 = nn.Conv2d(transformer_layer3_out_channels*2, target_decoder_size*2, 1, 1) # 320 -> 128
        #
        
        
        self.dec5_2 = cbr(transformer_layer4_out_channels*2,  512)
        self.dec5_1 = cbr(512,256)   
        
        self.dec4_2 = cbr(target_decoder_size*2+256, 
                          target_decoder_size*2+256)
        self.dec4_1 = cbr(target_decoder_size*2+256,
                          256)

        self.dec3_2 = cbr(target_decoder_size*2+256,
                          target_decoder_size*2+256)
        self.dec3_1 = cbr(target_decoder_size*2+256,
                          128)

        self.dec2_2 = cbr(target_decoder_size*2+128,
                          target_decoder_size*2+128)
        self.dec2_1 = cbr(target_decoder_size*2+128,
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
        trans0 = self.transformer_patch_emb(x) #352 16
        
        res1 = self.resnet_layer1(res0) #256   
        trans1 = self.transformer_layer1(trans0) #88 32
        res1,trans1 = self.mmtm1(res1,trans1)
        res1_down = self.cnn_down1(res1)        
        
        res2 = self.resnet_layer2(res1) #512
        trans2 = self.transformer_layer2(trans1) #44 64     
        res2,trans2 = self.mmtm2(res2,trans2)
        res2_down = self.cnn_down2(res2)   
        
        res3 = self.resnet_layer3(res2) #1024
        trans3 = self.transformer_layer3(trans2) #22 160
        res3,trans3 = self.mmtm3(res3,trans3)        
        res3_down = self.cnn_down3(res3)        
        
        res4 = self.resnet_layer4(res3) #2048

        trans4 = self.transformer_layer4(trans3) #11 256
        res4,trans4 = self.mmtm4(res4,trans4)
        res4_down = self.cnn_down4(res4)
        #enc_transformer = self.transformer_encoder(x)
        #print(torch.cat((enc_cnn,enc_transformer),1).shape)
        
        
        
        #print("tt")
        # print("trans0 : ", trans0.shape)
        # print("trans1 : ", trans1.shape)
        # print("trans2 : ", trans2.shape)
        # print("trans3 : ", trans3.shape)
        # print("trans4 : ", trans4.shape)
        
        

        ## bridge
        
        dec5_2 = self.dec5_2(torch.cat((res4_down,trans4),1))
        dec5_1 = self.dec5_1(dec5_2)
        
        ###
        
        
        unpool4 = F.interpolate(dec5_1, size=(trans3.shape[2],trans3.shape[3]), mode='bilinear', align_corners=True)
        
        dec4_skip = self.decoder_down3(torch.cat( (res3_down,trans3),1))
        dec4_2 = self.dec4_2(torch.cat( (dec4_skip,unpool4), 1) )        
        dec4_1 = self.dec4_1(dec4_2) 
        
        unpool3 = F.interpolate(dec4_1, size=(trans2.shape[2],trans2.shape[3]), mode='bilinear', align_corners=True)        
        dec3_skip = self.decoder_down2(torch.cat( (res2_down,trans2),1))
        dec3_2 = self.dec3_2(torch.cat((dec3_skip,unpool3),1)) 
        dec3_1 = self.dec3_1(dec3_2)  
        
        unpool2 = F.interpolate(dec3_1, size=(trans1.shape[2],trans1.shape[3]), mode='bilinear', align_corners=True)
        dec2_skip = self.decoder_down1(torch.cat( (res1_down,trans1),1))
        dec2_2 = self.dec2_2(torch.cat((dec2_skip, unpool2), 1)) 
        dec2_1 = self.dec2_1(dec2_2) 

        
        unpool1 = F.interpolate(dec2_1, size=(x.shape[2],x.shape[3]), mode='bilinear', align_corners=True)
        dec1_2 = self.dec1_2(unpool1)
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
        self.resnet_layer1 = resnet.layer1  # first blockc
        self.resnet_layer2 = resnet.layer2  # second blok
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
        
        
        self.cnn_down1 = nn.Conv2d(resnet_layer1_out_channels, transformer_layer1_out_channels,1,1)
        self.cnn_down2 = nn.Conv2d(resnet_layer2_out_channels, transformer_layer2_out_channels,1,1)
        self.cnn_down3 = nn.Conv2d(resnet_layer3_out_channels, transformer_layer3_out_channels,1,1)
        self.cnn_down4 = nn.Conv2d(resnet_layer4_out_channels, transformer_layer4_out_channels,1,1)
        
        del resnet; del transformer_encoder
        
        #### Multimodal Channel Attention
        
        self.mmtm1 = MMTM(resnet_layer1_out_channels, transformer_layer1_out_channels, 4)
        self.mmtm2 = MMTM(resnet_layer2_out_channels, transformer_layer2_out_channels, 4)
        self.mmtm3 = MMTM(resnet_layer3_out_channels, transformer_layer3_out_channels, 4)
        self.mmtm4 = MMTM(resnet_layer4_out_channels, transformer_layer4_out_channels, 4)        
        
        ####
        
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
        trans0 = self.transformer_patch_emb(x) #352 16
        
        res1 = self.resnet_layer1(res0) #256   
        trans1 = self.transformer_layer1(trans0) #88 32
        res1,trans1 = self.mmtm1(res1,trans1)
        res1_down = self.cnn_down1(res1)        
        
        res2 = self.resnet_layer2(res1) #512
        trans2 = self.transformer_layer2(trans1) #44 64
        import matplotlib.pyplot as plt
        for channel_idx in range(32):
            fig, ax = plt.subplots(1, 2)
            res_temp = res2[0][channel_idx].detach().cpu().numpy()
            res_temp = (res_temp - res_temp.min()) / (res_temp.max() - res_temp.min())
            
            trans_temp = trans2[0][channel_idx].detach().cpu().numpy()
            trans_temp = (trans_temp - trans_temp.min()) / (trans_temp.max() - trans_temp.min())
            
            ax[0].imshow(res_temp, cmap='jet', vmin=0, vmax=1)
            ax[1].imshow(trans_temp, cmap='jet', vmin=0, vmax=1)
            plt.savefig(f'../figure/before_{channel_idx}.png')     
        res2,trans2 = self.mmtm2(res2,trans2)
        for channel_idx in range(32):
            fig, ax = plt.subplots(1, 2)
            res_temp = res2[0][channel_idx].detach().cpu().numpy()
            res_temp = (res_temp - res_temp.min()) / (res_temp.max() - res_temp.min())
            
            trans_temp = trans2[0][channel_idx].detach().cpu().numpy()
            trans_temp = (trans_temp - trans_temp.min()) / (trans_temp.max() - trans_temp.min())
            
            ax[0].imshow(res_temp, cmap='jet', vmin=0, vmax=1)
            ax[1].imshow(trans_temp, cmap='jet', vmin=0, vmax=1)
            plt.savefig(f'../figure/after_{channel_idx}.png')
        res2_down = self.cnn_down2(res2)   
        
        res3 = self.resnet_layer3(res2) #1024
        trans3 = self.transformer_layer3(trans2) #22 160
        res3,trans3 = self.mmtm3(res3,trans3)        
        res3_down = self.cnn_down3(res3)        
        
        res4 = self.resnet_layer4(res3) #2048
        trans4 = self.transformer_layer4(trans3) #11 256
        res4,trans4 = self.mmtm4(res4,trans4)
        res4_down = self.cnn_down4(res4)
        #enc_transformer = self.transformer_encoder(x)
        #print(torch.cat((enc_cnn,enc_transformer),1).shape)
        
        
        
        #print("tt")
        # print("trans0 : ", trans0.shape)
        # print("trans1 : ", trans1.shape)
        # print("trans2 : ", trans2.shape)
        # print("trans3 : ", trans3.shape)
        # print("trans4 : ", trans4.shape)
        
        

        ## bridge
        
        dec5_2 = self.dec5_2(torch.cat((res4_down,trans4),1))
        dec5_1 = self.dec5_1(dec5_2)
        
        ###
        
        
        unpool4 = F.interpolate(dec5_1, size=(trans3.shape[2],trans3.shape[3]), mode='bilinear', align_corners=True)
        
        dec4_skip = torch.cat( (res3_down,trans3),1)
        dec4_2 = torch.cat( (dec4_skip,unpool4), 1)   
        dec4_1 = self.dec4_1(dec4_2) 
        
        unpool3 = F.interpolate(dec4_1, size=(trans2.shape[2],trans2.shape[3]), mode='bilinear', align_corners=True)        
        dec3_skip = torch.cat( (res2_down,trans2),1)
        dec3_2 = self.dec3_2(torch.cat((dec3_skip,unpool3),1)) 
        dec3_1 = self.dec3_1(dec3_2)  
        
        unpool2 = F.interpolate(dec3_1, size=(trans1.shape[2],trans1.shape[3]), mode='bilinear', align_corners=True)
        dec2_skip = torch.cat( (res1_down,trans1),1)
        dec2_2 = self.dec2_2(torch.cat((dec2_skip, unpool2), 1)) 
        dec2_1 = self.dec2_1(dec2_2) 

        
        unpool1 = F.interpolate(dec2_1, size=(x.shape[2],x.shape[3]), mode='bilinear', align_corners=True)
        dec1_2 = self.dec1_2(unpool1)
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
        self.resnet_layer1 = resnet.layer1  # first blockc
        self.resnet_layer2 = resnet.layer2  # second blok
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
        
        
        self.cnn_down1 = nn.Conv2d(resnet_layer1_out_channels, transformer_layer1_out_channels,1,1)
        self.cnn_down2 = nn.Conv2d(resnet_layer2_out_channels, transformer_layer2_out_channels,1,1)
        self.cnn_down3 = nn.Conv2d(resnet_layer3_out_channels, transformer_layer3_out_channels,1,1)
        self.cnn_down4 = nn.Conv2d(resnet_layer4_out_channels, transformer_layer4_out_channels,1,1)
        
        del resnet; del transformer_encoder
        
        #### Multimodal Channel Attention
        
        self.mmtm1 = MMTM(resnet_layer1_out_channels, transformer_layer1_out_channels, 4)
        self.mmtm2 = MMTM(resnet_layer2_out_channels, transformer_layer2_out_channels, 4)
        self.mmtm3 = MMTM(resnet_layer3_out_channels, transformer_layer3_out_channels, 4)
        self.mmtm4 = MMTM(resnet_layer4_out_channels, transformer_layer4_out_channels, 4)        
        
        ####
        
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
        trans0 = self.transformer_patch_emb(x) #352 16
        
        res1 = self.resnet_layer1(res0) #256   
        trans1 = self.transformer_layer1(trans0) #88 32
        res1,trans1 = self.mmtm1(res1,trans1)
        res1_down = self.cnn_down1(res1)        
        
        res2 = self.resnet_layer2(res1) #512
        trans2 = self.transformer_layer2(trans1) #44 64
        import matplotlib.pyplot as plt
        for channel_idx in range(32):
            fig, ax = plt.subplots(1, 2)
            res_temp = res2[0][channel_idx].detach().cpu().numpy()
            res_temp = (res_temp - res_temp.min()) / (res_temp.max() - res_temp.min())
            
            trans_temp = trans2[0][channel_idx].detach().cpu().numpy()
            trans_temp = (trans_temp - trans_temp.min()) / (trans_temp.max() - trans_temp.min())
            
            ax[0].imshow(res_temp, cmap='jet', vmin=0, vmax=1)
            ax[1].imshow(trans_temp, cmap='jet', vmin=0, vmax=1)
            plt.savefig(f'../figure/before_{channel_idx}.png')     
        res2,trans2 = self.mmtm2(res2,trans2)
        
        for channel_idx in range(32):
            fig, ax = plt.subplots(1, 2)
            res_temp = res2[0][channel_idx].detach().cpu().numpy()
            res_temp = (res_temp - res_temp.min()) / (res_temp.max() - res_temp.min())
            
            trans_temp = trans2[0][channel_idx].detach().cpu().numpy()
            trans_temp = (trans_temp - trans_temp.min()) / (trans_temp.max() - trans_temp.min())
            
            ax[0].imshow(res_temp, cmap='jet', vmin=0, vmax=1)
            ax[1].imshow(trans_temp, cmap='jet', vmin=0, vmax=1)
            plt.savefig(f'../figure/after_{channel_idx}.png')
        res2_down = self.cnn_down2(res2)   
        
        res3 = self.resnet_layer3(res2) #1024
        trans3 = self.transformer_layer3(trans2) #22 160
        res3,trans3 = self.mmtm3(res3,trans3)        
        res3_down = self.cnn_down3(res3)        
        
        res4 = self.resnet_layer4(res3) #2048
        trans4 = self.transformer_layer4(trans3) #11 256
        res4,trans4 = self.mmtm4(res4,trans4)
        res4_down = self.cnn_down4(res4)
        #enc_transformer = self.transformer_encoder(x)
        #print(torch.cat((enc_cnn,enc_transformer),1).shape)
        
        
        
        #print("tt")
        # print("trans0 : ", trans0.shape)
        # print("trans1 : ", trans1.shape)
        # print("trans2 : ", trans2.shape)
        # print("trans3 : ", trans3.shape)
        # print("trans4 : ", trans4.shape)
        
        

        ## bridge
        
        dec5_2 = self.dec5_2(torch.cat((res4_down,trans4),1))
        dec5_1 = self.dec5_1(dec5_2)
        
        ###
        
        
        unpool4 = F.interpolate(dec5_1, size=(trans3.shape[2],trans3.shape[3]), mode='bilinear', align_corners=True)
        
        dec4_skip = torch.cat( (res3_down,trans3),1)
        dec4_2 = torch.cat( (dec4_skip,unpool4), 1)   
        dec4_1 = self.dec4_1(dec4_2) 
        
        unpool3 = F.interpolate(dec4_1, size=(trans2.shape[2],trans2.shape[3]), mode='bilinear', align_corners=True)        
        dec3_skip = torch.cat( (res2_down,trans2),1)
        dec3_2 = self.dec3_2(torch.cat((dec3_skip,unpool3),1)) 
        dec3_1 = self.dec3_1(dec3_2)  
        
        unpool2 = F.interpolate(dec3_1, size=(trans1.shape[2],trans1.shape[3]), mode='bilinear', align_corners=True)
        dec2_skip = torch.cat( (res1_down,trans1),1)
        dec2_2 = self.dec2_2(torch.cat((dec2_skip, unpool2), 1)) 
        dec2_1 = self.dec2_1(dec2_2) 

        
        unpool1 = F.interpolate(dec2_1, size=(x.shape[2],x.shape[3]), mode='bilinear', align_corners=True)
        dec1_2 = self.dec1_2(unpool1)
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
        self.resnet_layer1 = resnet.layer1  # first blockc
        self.resnet_layer2 = resnet.layer2  # second blok
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
        
        
        self.cnn_down1 = nn.Conv2d(resnet_layer1_out_channels, transformer_layer1_out_channels,1,1)
        self.cnn_down2 = nn.Conv2d(resnet_layer2_out_channels, transformer_layer2_out_channels,1,1)
        self.cnn_down3 = nn.Conv2d(resnet_layer3_out_channels, transformer_layer3_out_channels,1,1)
        self.cnn_down4 = nn.Conv2d(resnet_layer4_out_channels, transformer_layer4_out_channels,1,1)
        
        del resnet; del transformer_encoder
        
        #### Multimodal Channel Attention
        
        self.mmtm1 = MMTM(resnet_layer1_out_channels, transformer_layer1_out_channels, 4)
        self.mmtm2 = MMTM(resnet_layer2_out_channels, transformer_layer2_out_channels, 4)
        self.mmtm3 = MMTM(resnet_layer3_out_channels, transformer_layer3_out_channels, 4)
        self.mmtm4 = MMTM(resnet_layer4_out_channels, transformer_layer4_out_channels, 4)        
        
        ####
        
        ### Spatial Attention
        self.sa1_cnn = SA(resnet_layer1_out_channels)
        self.sa2_cnn = SA(resnet_layer2_out_channels)
        self.sa3_cnn = SA(resnet_layer3_out_channels)
        self.sa4_cnn = SA(resnet_layer4_out_channels)
        
        self.sa1_trans = SA(transformer_layer1_out_channels)
        self.sa2_trans = SA(transformer_layer2_out_channels)
        self.sa3_trans = SA(transformer_layer3_out_channels)
        self.sa4_trans = SA(transformer_layer4_out_channels)
        
        ###
        
        
        
        
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
        
    def forward(self,x,visualize=False):
        #print(x.shape)
        
        #enc_cnn = self.cnn_encoder(x)
        
        res0 = self.resnet_layer0(x)
        trans0 = self.transformer_patch_emb(x) #352 16
        
        res1 = self.resnet_layer1(res0) #256   
        trans1 = self.transformer_layer1(trans0) #88 32
        if visualize:
            visualize_feature_map(res1,trans1, 'before_layer1')        
        
        res1,trans1 = self.mmtm1(res1,trans1)
        res1 = self.sa1_cnn(res1)
        #trans1 = self.sa1_trans(trans1)
        res1_down = self.cnn_down1(res1)        
        if visualize:
            visualize_feature_map(res1,trans1, 'after_layer1')
        
        res2 = self.resnet_layer2(res1) #512
        trans2 = self.transformer_layer2(trans1) #44 64
        res2,trans2 = self.mmtm2(res2,trans2)
        
        res2 = self.sa2_cnn(res2)
        #trans2 = self.sa2_trans(trans2)
        res2_down = self.cnn_down2(res2)   
        
        res3 = self.resnet_layer3(res2) #1024
        trans3 = self.transformer_layer3(trans2) #22 160
        res3,trans3 = self.mmtm3(res3,trans3)        
        
        res3 = self.sa3_cnn(res3)
        #trans3 = self.sa3_trans(trans3)
        res3_down = self.cnn_down3(res3)        
        
        res4 = self.resnet_layer4(res3) #2048
        trans4 = self.transformer_layer4(trans3) #11 256
        res4,trans4 = self.mmtm4(res4,trans4)
        
        res4 = self.sa4_cnn(res4)
        #trans4 = self.sa4_trans(trans4)
        res4_down = self.cnn_down4(res4)
        #enc_transformer = self.transformer_encoder(x)
        #print(torch.cat((enc_cnn,enc_transformer),1).shape)
        
        #print("tt")
        # print("trans0 : ", trans0.shape)
        # print("trans1 : ", trans1.shape)
        # print("trans2 : ", trans2.shape)
        # print("trans3 : ", trans3.shape)
        # print("trans4 : ", trans4.shape)
        
        

        ## bridge
        
        dec5_2 = self.dec5_2(torch.cat((res4_down,trans4),1))
        dec5_1 = self.dec5_1(dec5_2)
        
        ###
        
        
        unpool4 = F.interpolate(dec5_1, size=(trans3.shape[2],trans3.shape[3]), mode='bilinear', align_corners=True)
        
        dec4_skip = torch.cat( (res3_down,trans3),1)
        dec4_2 = torch.cat( (dec4_skip,unpool4), 1)   
        dec4_1 = self.dec4_1(dec4_2) 
        
        unpool3 = F.interpolate(dec4_1, size=(trans2.shape[2],trans2.shape[3]), mode='bilinear', align_corners=True)        
        dec3_skip = torch.cat( (res2_down,trans2),1)
        dec3_2 = self.dec3_2(torch.cat((dec3_skip,unpool3),1)) 
        dec3_1 = self.dec3_1(dec3_2)  
        
        unpool2 = F.interpolate(dec3_1, size=(trans1.shape[2],trans1.shape[3]), mode='bilinear', align_corners=True)
        dec2_skip = torch.cat( (res1_down,trans1),1)
        dec2_2 = self.dec2_2(torch.cat((dec2_skip, unpool2), 1)) 
        dec2_1 = self.dec2_1(dec2_2) 

        
        unpool1 = F.interpolate(dec2_1, size=(x.shape[2],x.shape[3]), mode='bilinear', align_corners=True)
        dec1_2 = self.dec1_2(unpool1)
        dec1_1 = self.dec1_1(dec1_2) 
        out = self.result(dec1_1)
        
        return out 