# import playsound
# import time
# def warring(warring_mp3_path):
#     try:
#         playsound.playsound(warring_mp3_path)
#     except:
#         print('播放失败！')
# if __name__ == '__main__':
#     while True:
#         warring('data/mp3/warring.mp3')
#         print(1)
#         # time.sleep(6)
# -*-coding:GBK -*-
import pygame

def play_music():
    file = r'data/mp3/warring.mp3'  # 要播放的歌曲本地地址
    pygame.mixer.init()  # mixer的初始化
    print("警报开启")  # 输出提示要播放的歌曲
    music = pygame.mixer.music.load(file)  # 载入一个音乐文件用于播放

    while True:
        # 检查音乐流播放，有返回True，没有返回False
        # 如果没有音乐流则选择播放
        if pygame.mixer.music.get_busy() == False:  # 检查是否正在播放音乐
            pygame.mixer.music.play()  # 开始播放音乐流
            print("return")