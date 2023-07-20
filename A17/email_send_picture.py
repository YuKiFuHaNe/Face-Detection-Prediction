import smtplib      # 实现发送邮件
import email        # 实现构造邮件格式和内容
# 负责构造文本
from email.mime.text import MIMEText
# 负责构造图片
from email.mime.image import MIMEImage
# 负责将多个对象集合起来
from email.mime.multipart import MIMEMultipart
from email.header import Header
class Send_Email:
    def __init__(self):
        self.mail_host = 'smtp.163.com'                 # SMTP服务器,这里使用163邮箱
        self.mail_license = 'GCKSUSDNBONOXHGY'          # 授权码
        self.mail_sender = 'b17876283049@163.com'       # 发件人邮箱
        self.mail_receivers = ['915228208@qq.com']      # 设置收件人邮箱地址
        self.mm = MIMEMultipart('related')

    def read_image_add(self,img_path,img_name):
        # 二进制读取图片
        image_data = open(img_path,'rb')
        # 设置读取获取的二进制数据
        message_image = MIMEImage(image_data.read())
        # 设置图片附件可直接浏览
        message_image.add_header('Content-ID', '<images1>')
        # 对文件重新命名（如果不加下面这行会在收件方显示乱码的bin文件）
        message_image["Content-Disposition"] = 'attachment; filename="{}'.format(img_name)
        # 关闭刚才打开的文件
        image_data.close()
        # 添加图片文件到邮件信息当中去
        self.mm.attach(message_image)

    def read_file_add(self,file_path,file_name):
        # 构造附件
        atta = MIMEText(open(file_path,'rb').read,'base64','utf-8')
        # 设置附件信息   对文件重新命名
        atta["Content-Disposition"] = 'attachment;filename="{}'.format(file_name)
        # 添加附件到邮件信息中去
        self.mm.attach(atta)

    def send_email(self):
        # 创建SMTP对象
        stp = smtplib.SMTP()
        # 设置发送人邮箱的域名的端口,端口地址为25
        stp.connect(self.mail_host)
        # set_debuglevel(1) 可以打印和SMTP服务器交互所有信息
        stp.set_debuglevel(1)
        # 登录邮箱，传递参数1：邮箱地址，参数2：邮箱授权码
        stp.login(self.mail_sender,self.mail_license)
        # 发送邮件，传递参数1：发件人邮箱地址，参数2：收件人邮箱地址，参数3：把邮件内容格式改为str
        stp.sendmail(self.mail_sender,self.mail_receivers,self.mm.as_string())
        print("邮件发送成功")
        # 关闭SMTP对象
        stp.quit()
    def run(self):
        # 设置邮件头部内容
        # 邮件主题
        subject_content = '''Python邮件测试'''
        # 设置返送着，注意遵守格式。里面邮箱为发件人邮箱
        self.mm['From'] = "sender_name<{}>".format(self.mail_sender)                # sender_name 可以是接收者收显示的名字  (可以自定义)
        # 设置接收者，注意严格遵守格式，里面为接收者邮箱
        self.mm["TO"] = "receiver_1_name<{}>".format(self.mail_receivers)           # "receiver_1_name<******@qq.com>,receiver_2_name<******@outlook.com>"    receiver_1_name接收者在收件人看到的  (可以自定义)
        # 设置邮箱主题
        self.mm["Subject"] = Header(subject_content,'utf-8')


        # 添加正文文本
        # 邮件正文
        body_content = """你好，这是一个测试邮件！"""
        # 构造文本,参数1：正文内容，参数2：文本格式，参数3：编码方式
        message_text = MIMEText(body_content,'plain','utf-8')
        # 向MINEMultipart对象添加文本对象
        self.mm.attach(message_text)

def main():
    em = Send_Email()
    em.run()
    em.send_email()


if __name__ == '__main__':
    main()