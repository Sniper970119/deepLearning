# -*- coding:utf-8 -*-

"""
      ┏┛ ┻━━━━━┛ ┻┓
      ┃　　　　　　 ┃
      ┃　　　━　　　┃
      ┃　┳┛　  ┗┳　┃
      ┃　　　　　　 ┃
      ┃　　　┻　　　┃
      ┃　　　　　　 ┃
      ┗━┓　　　┏━━━┛
        ┃　　　┃   神兽保佑
        ┃　　　┃   代码无BUG！
        ┃　　　┗━━━━━━━━━┓
        ┃　　　　　　　    ┣┓
        ┃　　　　         ┏┛
        ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
          ┃ ┫ ┫   ┃ ┫ ┫
          ┗━┻━┛   ┗━┻━┛
"""
import imageio
import PIL

setattr(PIL, 'PILLOW_VERSION', '2.8.0')
frames = []
for i in range(3, 121):
    # 计算需要在前面填充几个0
    num_fill = int(len("0000") - len(str(1))) + 1
    # 对索引进行填充
    filename = str(i).zfill(num_fill) + ".jpg"
    frames.append(imageio.imread('./out/' + filename))
print('read finish')
imageio.mimsave('result.gif', frames, 'GIF', duration=0.3)
