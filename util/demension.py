# input: (bs, c, l)     c = 1           (bs, 1, 24000)

# 第一层一维卷积
# (bs, mult * filters, l)
# (bs, 32, l)

# 残差块
# (bs, mult * filters, l)
# (bs, 32, l)

ratios = [2, 4, 5, 8]
mults = [1, 2, 4, 8]

# down1
# input: (bs, mult * filters, l)       => (bs, 32, 24000)
# out  : (bs, mult * filters * 2, l/2) => (bs, 64, 12000)

# down2
# input: (bs, mult * filters, l/2)     => (bs, 64, 12000)
# out  : (bs, mult * filters * 2, l/8) => (bs, 128, 3000)

# down3
# input: (bs, mult * filters, l/8)     => (bs, 128, 3000)
# out  : (bs, mult * filters * 2, l/40) => (bs, 256, 600)

# down4
#  input: (bs, mult * filters, l/40)    => (bs, 256, 600)
# out  : (bs, mult * filters * 2, l/320) => (bs, 512, 75)

# SLSTM
# 接收维度 = 8 * 32 = 512

# 最后一层
# input: (bs, 512, 75)
# out  : (bs, 128, 75)


##############################################################

# TCM module

# input: (bs, c ,l)

# encoder1
    # 两层 ConvTrans
    # 维度不变

    # (bs, c, l/2)
    
# encoder2
    # 两层 ConvTrans
    # 唯独不变
    
    # (bs, c, l/2)
    
    
## 更改维度

# 1 -> 2 -> 4 -> 5 -> 8
# (32,24k)->(64,12k)->(128,3k)->(256,600)->(512,75)

# 替换 s = 2, 5
# (32,24k)->*(64,12k)*->(128,3k)->*(256,600)*->(512,75)

# encoder1
    # (bs, )
    
    
    
    
  
# up-sample 
# SEAnetDecoder demension

# 一维卷积层
# input: (bs,128,l) -> (bs,128,75)
# out  : (bs,512,l) -> (bs,512,75)

# lstm
# 维度不变
# out  : (bs,512,75)

# up4
# change: (bs,512,75) -> (bs,256,600)

# up3
# change: (bs,256,600) -> (bs,128,3000)

# up2
# change: (bs,128,3000) -> (bs,64,12000)

# up1
# change: (bs,64,12000) -> (bs,32,24000)

# last
# (bs,32,24000) -> (bs,1,24000)




