# coding: utf-8
"""
    captcha.image
    ~~~~~~~~~~~~~

    Generate Image CAPTCHAs, just the normal image CAPTCHAs you are using.
"""

import os
import random
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
from io import BytesIO

import torchvision
from cvae import CVAE
from sample_cvae import DEVICE_STR, PREFERRED_DATATYPE
from utils import encode_label
import torch
try:
    from wheezy.captcha import image as wheezy_captcha
except ImportError:
    wheezy_captcha = None

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
DEFAULT_FONTS = [os.path.join(DATA_DIR, 'DroidSansMono-V4.ttf')]

if wheezy_captcha:
    __all__ = ['ImageCaptcha', 'WheezyCaptcha']
else:
    __all__ = ['ImageCaptcha']


table  =  []
for  i  in  range( 256 ):
    table.append( int(i * 1.97) )


class _Captcha(object):
    def generate(self, chars, format='png', boxes=False):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im, offset_points = self.generate_image(chars, boxes)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out, offset_points

    def write(self, chars, output, format='png', boxes=False):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        im, offset_points = self.generate_image(chars, boxes)
        im.save(output, format=format)
        return offset_points

    def write_letter(self, letter, output, format='png') -> None:
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        color = (1, 1, 1)
        background = (255, 255, 255)
        self._width = 30
        self._height = 60
        im = self.create_captcha_letter(letter, color, background)
        im.save(output, format=format)

    def write_with_cvae(self, chars, output, model: CVAE, format='png'):
        im, _ = self.generate_image_with_cvae(chars, model)
        im.save(output, format=format)
        return None


class WheezyCaptcha(_Captcha):
    """Create an image CAPTCHA with wheezy.captcha."""
    def __init__(self, width=200, height=75, fonts=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS

    def generate_image(self, chars):
        text_drawings = [
            wheezy_captcha.warp(),
            wheezy_captcha.rotate(),
            wheezy_captcha.offset(),
        ]
        fn = wheezy_captcha.captcha(
            drawings=[
                wheezy_captcha.background(),
                wheezy_captcha.text(fonts=self._fonts, drawings=text_drawings),
                wheezy_captcha.curve(),
                wheezy_captcha.noise(),
                wheezy_captcha.smooth(),
            ],
            width=self._width,
            height=self._height,
        )
        return fn(chars)


class ImageCaptcha(_Captcha):
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """
    def __init__(self, width=160, height=60, fonts=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts = []

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    @staticmethod
    def create_noise_curve(image, color):
        w, h = image.size
        x1 = random.randint(0, int(w / 5))
        x2 = random.randint(w - int(w / 5), w)
        y1 = random.randint(int(h / 5), h - int(h / 5))
        y2 = random.randint(y1, h - int(h / 5))
        points = [x1, y1, x2, y2]
        end = random.randint(160, 200)
        start = random.randint(0, 20)
        Draw(image).arc(points, start, end, fill=color)
        return image

    @staticmethod
    def create_noise_dots(image, color, width=3, number=30):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image

    def create_captcha_letter(self, letter, color, background):
        """Create a single CAPTCHA letter.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        
        Optimaly this would be possible by simply passing a list containing a
        single letter to the create_captcha_image method, however, it didn't
        work so creating a second method was easier.
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = Image.new('RGBA', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            return im

        im = _draw_character(letter)
        im = im.resize((self._width, self._height))
        image.paste(im, mask=im.split()[3])

        return image

    def create_captcha_image_with_cvae(self, chars, color, background, model: CVAE):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)
        to_pil = torchvision.transforms.ToPILImage()

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            
            if c == " ":
                im = Image.new('RGB', (w + dx, h + dy))
                Draw(im).text((dx, dy), c, font=font, fill=color)
            else:
                encoded_label = encode_label(str(c))
                label = torch.ones((1,1)) * encoded_label
                label = label.to(torch.int64).to(DEVICE_STR)  # labels can be int
                im = model.sampling(n=1, c=label)
                im = to_pil(im[0])

            # Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            '''
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            '''
            return im

        images = []
        is_space = []
        for c in chars:
            if random.random() > 0.5 and len(chars) > 1:  # skip for only one
                images.append(_draw_character(" "))
                is_space.append(True)
            images.append(_draw_character(c))
            is_space.append(False)

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        offset_points = []

        for idx, im in enumerate(images):
            w, h = im.size
            mask = im.convert('L').point(table)
            upper = int((self._height - h) / 2)
            image.paste(im, (offset, upper), mask)
            left = offset
            if(not is_space[idx]):
                offset_points.append(((max(left,0), max(upper, 0)),(min(left+w, width-1),min(upper+h, self._height-1))))
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            ratio = self._width / width
            for idx in range(len(offset_points)):
                offset_points[idx] = tuple([(min(int(float(w)*ratio),self._width-1),h) for w,h in offset_points[idx]])
            image = image.resize((self._width, self._height))

        # breakpoint()
        image.save("/tmp/out.png")
        return image, offset_points

    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = Image.new('RGBA', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            return im

        images = []
        is_space = []
        for c in chars:
            if random.random() > 0.5 and len(chars) > 1:  # skip for only one
                images.append(_draw_character(" "))
                is_space.append(True)
            images.append(_draw_character(c))
            is_space.append(False)

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        offset_points = []

        for idx, im in enumerate(images):
            w, h = im.size
            # mask = im.convert('L').point(table)
            upper = int((self._height - h) / 2)
            # image.paste(im, (offset, upper), mask)
            image.paste(im, (offset, upper))
            left = offset
            if(not is_space[idx]):
                offset_points.append(((max(left,0), max(upper, 0)),(min(left+w, width-1),min(upper+h, self._height-1))))
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            ratio = self._width / width
            for idx in range(len(offset_points)):
                offset_points[idx] = tuple([(min(int(float(w)*ratio),self._width-1),h) for w,h in offset_points[idx]])
            image = image.resize((self._width, self._height))

        return image, offset_points

    def generate_image(self, chars, boxes=False):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = random_color(238, 255)
        color = random_color(10, 200, random.randint(220, 255))
        im, offset_points = self.create_captcha_image(chars, color, background)
        self.create_noise_dots(im, color)
        self.create_noise_curve(im, color)
        im = im.filter(ImageFilter.SMOOTH)
        if boxes:
            draw = Draw(im)
            for points in offset_points:
                draw.rectangle(points, outline="red")
        return im, offset_points

    def generate_image_with_cvae(self, chars, model):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = random_color(238, 255)
        color = random_color(10, 200, random.randint(220, 255))
        im, _ = self.create_captcha_image_with_cvae(chars, color, background, model)
        self.create_noise_dots(im, color)
        self.create_noise_curve(im, color)
        im = im.filter(ImageFilter.SMOOTH)
        return im, None


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)
