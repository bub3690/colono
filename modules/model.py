import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm




# https://velog.io/@pre_f_86/UNET-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0



def visualize_feature_map(feature_map1,feature_map2, name):
        import matplotlib.pyplot as plt
        for channel_idx in range(32):
            fig, ax = plt.subplots(1, 2)
            res_temp = feature_map1[0][channel_idx].detach().cpu().numpy()
            res_temp = (res_temp - res_temp.min()) / (res_temp.max() - res_temp.min())
            
            trans_temp = feature_map2[0][channel_idx].detach().cpu().numpy()
            trans_temp = (trans_temp - trans_temp.min()) / (trans_temp.max() - trans_temp.min())
            
            ax[0].imshow(res_temp, cmap='jet', vmin=0, vmax=1)
            ax[1].imshow(trans_temp, cmap='jet', vmin=0, vmax=1)
            plt.savefig(f'../figure/{name}_{channel_idx}.png')     


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



class MMTM(nn.Module):
    def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # initialize
        # with torch.no_grad():
        #   self.fc_squeeze.apply(init_weights)
        #   self.fc_visual.apply(init_weights)
        #   self.fc_skeleton.apply(init_weights)    

    def forward(self, visual, skeleton):
        squeeze_array = []
        for tensor in [visual, skeleton]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)
    
        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out, skeleton * sk_out


class SA(nn.Module):
    def __init__(self, dim_feature):
        super(SA, self).__init__()
        
        self.cnn_conv = nn.Conv2d(dim_feature, 1, 1)
        self.sigmoid = nn.Sigmoid()

        # learnable param, alpha and beta for negative att, positive att
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
        
        self.refinement_conv = nn.Conv2d(dim_feature, dim_feature, 3,1,1) # non linear함을 추가함.
        
        # initialize
        # with torch.no_grad():
        #   self.fc_squeeze.apply(init_weights)
        #   self.fc_visual.apply(init_weights)
        #   self.fc_skeleton.apply(init_weights)    

    def forward(self, input_feature):
        
        spatial_att = self.cnn_conv(input_feature)
        spatial_att = self.sigmoid(spatial_att)

        negative_att = (1 - spatial_att) * self.alpha
        positive_att = spatial_att * self.beta
        
        input_feature = input_feature*positive_att + input_feature*negative_att
        input_feature = self.refinement_conv(input_feature)
        
        return input_feature





class MMTM_max(nn.Module):
    def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM_max, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim*2, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # initialize
    # with torch.no_grad():
    #   self.fc_squeeze.apply(init_weights)
    #   self.fc_visual.apply(init_weights)
    #   self.fc_skeleton.apply(init_weights)

    def forward(self, visual, skeleton):
        squeeze_array = []
        for tensor in [visual, skeleton]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
            squeeze_array.append(torch.max(tview, dim=-1)[0])
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out, skeleton * sk_out



class LinearSRA(nn.Module):
    def __init__(self, cnn_channels,trans_channels, pooling_size=7, heads=8):
        super(LinearSRA, self).__init__()
        self.pooling_size = pooling_size
        self.heads = heads
        
        self.cnn_channels = cnn_channels
        
        self.channels = trans_channels
        
        # MultiheadAttention 요구 사항에 맞춰 feature 크기 조정
        self.attention = nn.MultiheadAttention(self.channels, heads)
        
        # Linear layers to transform the input features into query, key, and value
        # 여기서 채널 다운 발생.
        self.to_query = nn.Linear(self.cnn_channels, self.channels)
        self.to_key = nn.Linear(self.cnn_channels, self.channels)
        
        self.to_value = nn.Linear(self.channels, self.channels)
        
        #up channel for cnn channel
        self.up_cnn = nn.Linear(self.channels, self.cnn_channels)
        
        # sigomoid
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x_cnn, x_trans):
        # cnn은 linear layer로 channel이 줄여진다.
        # x: 입력 피처 맵 (batch_size, channels, height, width)
        
        # 평균 풀링을 사용하여 공간 차원 축소
        x_cnn_pooled = F.adaptive_avg_pool2d(x_cnn, (self.pooling_size, self.pooling_size))
        x_trans_pooled = F.adaptive_avg_pool2d(x_trans, (self.pooling_size, self.pooling_size))
        
        # (batch_size, channels, pooling_size, pooling_size) -> (batch_size, pooling_size*pooling_size, channels)
        x_cnn_pooled = x_cnn_pooled.view(x_cnn_pooled.size(0), self.cnn_channels, -1).permute(0, 2, 1)
        x_trans_pooled = x_trans_pooled.view(x_trans_pooled.size(0), self.channels, -1).permute(0, 2, 1)
        
        # Query, Key, Value 생성
        query = self.to_query(x_cnn_pooled)
        key = self.to_key(x_cnn_pooled)
        value = self.to_value(x_trans_pooled)
        
        # Attention 연산 수행
        attn_output, _ = self.attention(query, key, value)
        cnn_result = self.up_cnn(attn_output)

        
        
        cnn_result = cnn_result.permute(0, 2, 1).view(x_cnn.size(0), self.cnn_channels, self.pooling_size, self.pooling_size)

        #upsample
        cnn_result = F.interpolate(cnn_result, size=(x_cnn.size(2),x_cnn.size(3)), mode='bilinear', align_corners=True)
        transform_result = attn_output.permute(0, 2, 1).view(x_trans.size(0), self.channels, self.pooling_size, self.pooling_size)
        transform_result = F.interpolate(transform_result, size=(x_trans.size(2), x_trans.size(3)), mode='bilinear', align_corners=True)
        
        cnn_result = self.sigmoid(cnn_result)
        transform_result = self.sigmoid(transform_result)
        
        return cnn_result, transform_result


class LinearSRA_channel(nn.Module):
    def __init__(self, cnn_channels,trans_channels, pooling_size=1, heads=8):
        super(LinearSRA_channel, self).__init__()
        self.pooling_size = pooling_size
        self.heads = heads
        
        self.cnn_channels = cnn_channels
        
        self.channels = trans_channels
        
        # MultiheadAttention 요구 사항에 맞춰 feature 크기 조정
        self.attention = nn.MultiheadAttention(self.channels, heads)
        
        # Linear layers to transform the input features into query, key, and value
        # 여기서 채널 다운 발생.
        self.to_query = nn.Linear(self.cnn_channels, self.channels)
        self.to_key = nn.Linear(self.cnn_channels, self.channels)
        
        self.to_value = nn.Linear(self.channels, self.channels)
        
        #up channel for cnn channel
        self.up_cnn = nn.Linear(self.channels, self.cnn_channels)
        
        # sigomoid
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x_cnn, x_trans):
        # cnn은 linear layer로 channel이 줄여진다.
        # x: 입력 피처 맵 (batch_size, channels, height, width)
        
        # 평균 풀링을 사용하여 공간 차원 축소
        x_cnn_pooled = F.adaptive_avg_pool2d(x_cnn, (self.pooling_size, self.pooling_size))
        x_trans_pooled = F.adaptive_avg_pool2d(x_trans, (self.pooling_size, self.pooling_size))
        
        # (batch_size, channels, pooling_size, pooling_size) -> (batch_size, pooling_size*pooling_size, channels)
        x_cnn_pooled = x_cnn_pooled.view(x_cnn_pooled.size(0), self.cnn_channels, -1).permute(0, 2, 1)
        x_trans_pooled = x_trans_pooled.view(x_trans_pooled.size(0), self.channels, -1).permute(0, 2, 1)
        
        # Query, Key, Value 생성
        query = self.to_query(x_cnn_pooled)
        key = self.to_key(x_cnn_pooled)
        value = self.to_value(x_trans_pooled)
        
        # Attention 연산 수행
        attn_output, _ = self.attention(query, key, value) 
        # attn_output의미 : 해당 채널의 중요도를 나타냄. query와 key의 내적을 통해 나온 값. 
        # 그 후 value와 어떤 연산을
        cnn_result = self.up_cnn(attn_output)

        
        
        cnn_result = cnn_result.permute(0, 2, 1).view(x_cnn.size(0), self.cnn_channels, self.pooling_size, self.pooling_size)
        transform_result = attn_output.permute(0, 2, 1).view(x_trans.size(0), self.channels, self.pooling_size, self.pooling_size)
        
        cnn_result = self.sigmoid(cnn_result)
        transform_result = self.sigmoid(transform_result)
        
        
        return cnn_result, transform_result


#############

class Hybrid2(nn.Module):
    def __init__(self):
        super(Hybrid2,self).__init__()
        
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
        
        #### Attention
        
        self.lsra1 = LinearSRA(resnet_layer1_out_channels,transformer_layer1_out_channels)
        self.lsra2 = LinearSRA(resnet_layer2_out_channels,transformer_layer2_out_channels)
        self.lsra3 = LinearSRA(resnet_layer3_out_channels,transformer_layer3_out_channels)
        self.lsra4 = LinearSRA(resnet_layer4_out_channels,transformer_layer4_out_channels)
        
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
        
    def forward(self,x,visualize=False):
        #print(x.shape)
        
        #enc_cnn = self.cnn_encoder(x)
        
        res0 = self.resnet_layer0(x)
        trans0 = self.transformer_patch_emb(x) #352 16
        
        res1 = self.resnet_layer1(res0) #256   
        trans1 = self.transformer_layer1(trans0) #88 32
        if visualize:
            visualize_feature_map(res1,trans1, 'before_layer1')        
        
        res1_att, trans1_att = self.lsra1(res1,trans1)
        res1 = res1*res1_att
        trans1 = trans1*trans1_att
        
        res1_down = self.cnn_down1(res1)        
        if visualize:
            visualize_feature_map(res1,trans1, 'after_layer1')
        
        res2 = self.resnet_layer2(res1) #512
        trans2 = self.transformer_layer2(trans1) #44 64
        
        res2_att, trans2_att = self.lsra2(res2,trans2)
        res2 = res2*res2_att
        trans2 = trans2*trans2_att
        
        res2_down = self.cnn_down2(res2)   
        
        res3 = self.resnet_layer3(res2) #1024
        trans3 = self.transformer_layer3(trans2) #22 160     
        
        res3_att, trans3_att = self.lsra3(res3,trans3)
        res3 = res3*res3_att
        trans3 = trans3*trans3_att
        
        res3_down = self.cnn_down3(res3)        
        
        res4 = self.resnet_layer4(res3) #2048
        trans4 = self.transformer_layer4(trans3) #11 256
        
        res4_att, trans4_att = self.lsra4(res4,trans4)
        res4 = res4*res4_att
        trans4 = trans4*trans4_att
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
        
        
        
        #resnet = timm.create_model('resnet50', pretrained=True)
        resnet = timm.create_model('resnext50_32x4d', pretrained=True)
        #self.cnn_encoder = nn.Sequential(*list(self.cnn_encoder.children())[:-2]) # 2048,11,11
        

        # Split ResNet50 into separate blocks for skip connections
        print(resnet)
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





def model_loader(config):
    if config.model == 'unet':
        model = UNet()
    elif config.model == 'hybrid':
        model = Hybrid() # target
    elif config.model == 'hybrid2':
        model = Hybrid2()
    else:
        model = UNet()
    return model


if __name__ == '__main__':
    pass
    #model = UNet()
    #model = Hybrid()
    #print(model)
    
    img_path = "/ssd2/colono/data/TestDataset/CVC-300/images/149.png"
    
    img = torch.randn(1,3,352,352)
    #img = torchvision.io.read_image(img_path).unsqueeze(0).float() / 255.0 # 
    
    sample_model = Hybrid()
    sample_model.eval()
    #sample_model.load_state_dict(torch.load('/ssd2/colono/checkpoint/hybrid_99.pth'))
    #sample_model = Hybrid2() 
    #sample_model.load_state_dict(torch.load('/ssd2/colono/checkpoint/hybrid2_99.pth'))
    
    sample_img = F.interpolate(img, size=(352,352), mode='bilinear', align_corners=True)
    #sample_img = F.interpolate(img, scale_factor=1.0, mode='bilinear', align_corners=True)

    sample_output = sample_model(sample_img, visualize=False)
    print(sample_output.shape)    
        
        