import ftplib
from ftplib import FTP
import os
import logging
import datetime
class ftp:
    def __init__(self):
        self.host = "101.33.204.87"
        self.port = 21
        self.user = "ftpadmin"
        self.password = "@zhongpeng123"
        self.ftp = FTP()
        self.bufsize = 4096
    def ftp_connect(self):
        '''
        FTP连接
        :return:
        '''
        self.ftp.connect(host=self.host,port=self.port)
        self.ftp.login(user=self.user,passwd=self.password)
    def print(self):
        print(self.ftp.dir())
        print(self.ftp.nlst())
    def quit(self):
        self.ftp.quit()
    def down_data(self,loacl_filename,cloud_filename):
        '''
        下载文件到本地
        :param loacl_filename:本地文件
        :param cloud_filename:云端文件
        '''
        try:
            fp = open(loacl_filename,mode='wb')
            self.ftp.retrbinary('RETR '+cloud_filename,fp.write,self.bufsize)
            fp.close()
            # print("下载完成")
            logging.info('%-40s / %-10s',f'下载完成 / Download completed:{cloud_filename}',datetime.datetime.now())
        except:
            logging.info('%-40s / %-10s', f'下载失败 / Download failed:{cloud_filename}', datetime.datetime.now())
    def up_data(self,loacl_filename,cloud_filename):
        '''
        上传文件到云端
        :param loacl_filename: 本地文件
        :param cloud_filename: 云端文件
        :return:
        '''
        try:
            fp = open(loacl_filename, mode='rb')
            self.ftp.storbinary('STOR ' + cloud_filename, fp, self.bufsize)
            fp.close()
            # print("上传完成")
            logging.info('%-40s / %-10s', f'上传完成 / Upload completed:{loacl_filename}', datetime.datetime.now())
        except:
            logging.error('%-40s / %-10s', f'上传失败 / Upload failed:{loacl_filename}', datetime.datetime.now())
    def double_file_path(self,all_path,str_index):
        '''
        对多重路径云端进行检查
        :param all_path: 所有路径 / str
        :return: 返回遍历检查并创建的路径
        '''
        all_path =str(all_path)
        name = str_index
        path = all_path.split(str_index,maxsplit=1)[0]
        path = path.strip('./')
        for i in range(path.count('/')+1):
            try:
                p = path.split('/')[:i+1]
                # print(p)
                self.ftp.mkd('/'.join(p))
                # print('/'.join(p))
            except:
                continue
        return path  # data/recording/*/2023/03/29/*.jpg
if __name__ == '__main__':
    '''
    ./data/recording/Unknow/{}/{}/{}/Unknow{}_{}_{}.jpg
    '''
    ftp1 = ftp()
    ftp1.ftp_connect()
    ftp1.print()
    # ftp1.down_data('Double_loss_vloss.png','Double_loss_vloss.png')
    # ftp1.up_data('Web.rar','Web.rar')
    path = ftp1.double_file_path('./data/recording/Unknow/2023/3/3/Unknow4_mask_10_56.jpg','Unknow01_mask_12.jpg')
    ftp1.up_data('./data/recording/Unknow/2023/3/3/Unknow4_mask_10_56.jpg', path+'/'+'Unknow4_mask_10_56.jpg')
    ftp1.quit()