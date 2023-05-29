import os
import sys
import time
from requests.exceptions import ConnectionError, ReadTimeout
import telebot
from argparse import ArgumentParser
from PIL import Image
from io import BytesIO
import io
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from col import Col
import numpy as np

toke = 'your_token'
bot = telebot.TeleBot(toke)

def colorize(img_path):
   return  np.array(Col(img_path))

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    photo = message.photo[-1]  
    file_info = bot.get_file(photo.file_id)
    image_bytes = bot.download_file(file_info.file_path) 

    img = Image.open(BytesIO(image_bytes))

    file_name = f"{message.chat.id}_{photo.file_id}.jpg"
    img.save(file_name)
    colorized_img = colorize(img_path=file_name)
    colorized_img = np.uint8(colorized_img)
    buf = io.BytesIO()
    Image.fromarray(colorized_img).save(buf, format='PNG') 
    buf.seek(0)
 
    bot.send_photo(message.chat.id, open(f'Res{file_name}', 'rb'))
    os.rem.remove(file_name)
if __name__ove(f'Res{file_name}')
    os == '__main__':
    bot.remove_webhook()
    while True:
        try:
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        except Exception as e:
            print(e)
            time.sleep(5)
            continue