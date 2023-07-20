import pymysql
import logging
import datetime
class mysql():
    def __init__(self):
        self.host = '101.33.204.87'
        self.user = 'A17'
        self.password = 'zhongpeng'
        self.database='A17'
        self.charset = 'utf8'
        self.port = 3306
        self.image_r = None               # 图片
        self.image_roi_face_r = None      # 图片roi
        self.image_mask_r = None          # 图片mask
        self.name = None                  # 用户名称
        self.flag = 0
    def connect_mysql(self):
        try:
            self.conn = pymysql.connect(host=self.host,
                                 user=self.user,
                                 passwd=self.password,
                                 database=self.database,
                                 charset=self.charset,
                                 port=self.port
                                 )
            self.flag = 1
        except:
                self.flag = 0
        print(self.flag)
    def load_name(self,name):
        '''
        读取名称
        '''
        self.name = name

    def load_img(self,img_path,img_mask_path,img_roi_path):
        '''
        二进制读取文件
        :param img_path: str
        :param img_mask_path:str
        :param img_roi_path: str
        '''
        fp1 = open(img_path,'rb')
        fp2 = open(img_mask_path,'rb')
        fp3 = open(img_roi_path,'rb')
        # 将图片转换成2进制文件
        self.image_r = fp1.read()
        self.image_mask_r = fp2.read()
        self.image_roi_face_r = fp3.read()
        # 关闭文件
        fp1.close(),fp2.close(),fp3.close()

    def load_path(self,name,img_path,img_mask_path,img_roi_path):
        self.name = name
        self.image_r = img_path
        self.image_mask_r = img_mask_path
        self.image_roi_face_r = img_roi_path

    def post_mysql_data(self):
        self.connect_mysql()
        if self.flag:
            print(self.image_r)
            print(self.image_mask_r)
            print(self.image_roi_face_r)
            print(self.name)
            # print(str(self.image_roi_face_r))
            # self.image_r = (str(self.image_r)).strip('b')
            try:
                cursor = self.conn.cursor()
                # sql = f'insert into A17.a17_data (time,user,photo_all,photo_mask,photo_face) VALUES (%s,%s,%s,%s,%s);'
                sql = f'''insert into A17.a17_data (time, user, photo_all, photo_mask, photo_face) values (now(),"{self.name}","{self.image_r}","{self.image_mask_r}","{self.image_roi_face_r}");'''
                # args = ('now()', f"{self.name}",self.image_r,self.image_mask_r,self.image_roi_face_r)
                # args = ('now()', '1', None, None, None)
                result = cursor.execute(sql)
                print(result)
                self.conn.commit()
                cursor.close()
                self.conn.close()
                print("数据上传成功")
                logging.info('%-40s / %-10s','数据上传成功 / Data uploaded successfully',datetime.datetime.now())
            except:
                print("数据上传失败")
                logging.error('%-40% / %-10s', '数据上传失败 / Data uploaded failed', datetime.datetime.now())
        else:
            print("数据库连接失败")
            logging.error('%-40% / %-10s', '数据库连接失败 / Database connection failed', datetime.datetime.now())

    def select_all(self):
        '''
        查询所有记录
        :return:list
        '''
        self.connect_mysql()
        if self.flag:
            # print(str(self.image_roi_face_r))
            # self.image_r = (str(self.image_r)).strip('b')
            try:
                cursor = self.conn.cursor()
                # sql = f'insert into A17.a17_data (time,user,photo_all,photo_mask,photo_face) VALUES (%s,%s,%s,%s,%s);'
                sql = f'''select * from A17.a17_data order by time desc '''
                result = cursor.execute(sql)

                res = cursor.fetchall()
                print(res)
                # self.conn.commit()
                cursor.close()
                self.conn.close()
                print("数据查询成功")
                return res
            except:
                print("数据查询失败")
                return ()
        else:
            print("数据库连接失败")
            return ()
    def select_date(self,choose_date):
        '''
               查询指定日期记录
               :return:list
               '''
        self.connect_mysql()
        if self.flag:
            # print(str(self.image_roi_face_r))
            # self.image_r = (str(self.image_r)).strip('b')
            try:
                cursor = self.conn.cursor()
                # sql = f'insert into A17.a17_data (time,user,photo_all,photo_mask,photo_face) VALUES (%s,%s,%s,%s,%s);'
                sql = f'''select * from A17.a17_data where DATE_FORMAT(time ,'%Y-%m-%d')="{choose_date}" order by time desc '''
                result = cursor.execute(sql)

                res = cursor.fetchall()
                print(res)
                # self.conn.commit()
                cursor.close()
                self.conn.close()
                print("数据查询成功")
                return res
            except:
                print("数据查询失败")
                return ()
        else:
            print("数据库连接失败")
            return ()

'''
# # 获取返回结果
    # res = cursor.fetchall()  # 获取所有记录列表
    # # res = cursor.fetchone()  # 获取单条数据.
    # # res = cursor.fetchmany(3)  # 列表套字典 几个数据
'''
if __name__ == '__main__':
    my = mysql()
    my.load_name('1')
    my.load_img('./data/recording/use/2023/3/2/use0_22_41.jpg','./data/recording/use/2023/3/2/use0_mask_22_41.jpg','./data/recording/use/2023/3/2/use0_ROI_22_41.jpg')
    my.post_mysql_data()