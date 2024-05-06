import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        return x



class LinearSRA(nn.Module):
    def __init__(self, channels, pooling_size=7, heads=8):
        super(LinearSRA, self).__init__()
        self.pooling_size = pooling_size
        self.heads = heads
        self.channels = channels
        
        # MultiheadAttention 요구 사항에 맞춰 feature 크기 조정
        self.attention = nn.MultiheadAttention(channels, heads)
        
        # Linear layers to transform the input features into query, key, and value
        self.to_query = nn.Linear(channels, channels)
        self.to_key = nn.Linear(channels, channels)
        self.to_value = nn.Linear(channels, channels)

    def forward(self, x):
        # x: 입력 피처 맵 (batch_size, channels, height, width)
        
        # 평균 풀링을 사용하여 공간 차원 축소
        x_pooled = F.adaptive_avg_pool2d(x, (self.pooling_size, self.pooling_size))
        print(x_pooled.shape)
        
        # (batch_size, channels, pooling_size, pooling_size) -> (batch_size, pooling_size*pooling_size, channels)
        x_pooled = x_pooled.view(x_pooled.size(0), self.channels, -1).permute(0, 2, 1)
        print(x_pooled.shape)
        
        # Query, Key, Value 생성
        query = self.to_query(x_pooled)
        key = self.to_key(x_pooled)
        value = self.to_value(x_pooled)
        
        # Attention 연산 수행
        attn_output, _ = self.attention(query, key, value)
        return attn_output


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
        
        return cnn_result, transform_result



# 예제 입력
batch_size = 1
channels = 3
#height, width = 352, 352
#dummy_img = torch.randn(batch_size, channels, height, width)

#simplecnn =SimpleCNN()
#dummy_feature = simplecnn(dummy_img)


cnn_channels=128
trans_channels = 80

dummy_feature1 = torch.randn(batch_size, 128, 128, 128)
dummy_feature2 = torch.randn(batch_size, 80, 128, 128)
print(dummy_feature1.shape)  # 출력 차원 확인
print(dummy_feature2.shape)  # 출력 차원 확인

model = LinearSRA(cnn_channels,trans_channels, pooling_size=7, heads=8)
cnn_result,transform_result = model(dummy_feature1,dummy_feature2)
print(cnn_result.shape, transform_result.shape)  # 출력 차원 확인