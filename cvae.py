'''
Different approaches to train the cvae

We need some form of training input:
    - whole captcha = image and string:
        - Allows to train on different captcha layouts as we do not need any
          bounding boxes compared to the "solving captchas" task
        - we could also train on the mnist dataset and use the generated
          letters to generate captachs similar to our algorithmic approach
            - generated images are hybrid machine learning and algorithmic:
        - instead of the mnist dataset we might use the different fonts and to
          generate simple letters and encode trying to achieve encoding the
          font type into the latent space allowing us to create letters that
          resemble a combination or merging of different fonts, which might
          make it harder for an classification network to correctly classify a
          letter
            - we should still only encode the letter not the font, the letter
              is a hard requirement that has to be met while the font can be
              something where a hard assignment to any fontset isn't required



Different CVAEs:
1. CVAE generating a whole captcha:
    - generating a whole captcha is a bit annoying because we have such a high
      class number:
classes = 26*2 (letters) + 10 (numbers) + 1 (space/emptiness)

we want to make captchas with e.g. max 6 numbers
63^6 -> 62523502209 (problematic)
Problem:
! For now let's just consider the first approach a bit more feasible

Problem is we need something that can produce the thing based on a a label
because the human has to assign the generated image to a class.

2. CVAE genearting only one letter at the time:
    - we need single letter but best already distorted
        - > adapt the data generation script

Broader Prolbems that should be considered:
    - What happens if the captcha is fillped, can we just rotate it mulitple
      times (steps of 20 deg) and then use the argmax(sum(score)) For this to
      work we would have to train with slightly augmented data (also rotate
      training data a little bit to make the deg window that we have to hit
      more likely)

'''
# train loop
