from captcha.image import ImageCaptcha
import random, string

image = ImageCaptcha()

def generate_python_captcha(number_of_chars: int = 0):
    if(number_of_chars == 0):
        number_of_chars = random.randrange(2)+4
    content = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(number_of_chars)) # take uppercase only to reduce similar chars like w and W
    #data = image.generate(content)
    image.write(content, f"generated_captchas/imagecaptcha/{content}.png")

def generate_own_captcha():
    pass

if __name__ == "__main__":
    for i in range(10):
        generate_python_captcha()
