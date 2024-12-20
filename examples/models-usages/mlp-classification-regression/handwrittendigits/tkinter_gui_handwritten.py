import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw

from neuralnetlib.models import Sequential


def main():
    model = Sequential.load('my_mnist_model.npz')
    window = tk.Tk()
    window.geometry("480x480")
    canvas = tk.Canvas(window, width=280, height=280, bg='black')
    canvas.pack()
    label = tk.Label(window, text="Draw a digit", font=("Helvetica", 20))
    label.pack()

    button_clear = tk.Button(window, text="Clear", command=lambda: clear(canvas))
    button_clear.pack()

    # Removed the Predict button as prediction is now automatic

    def clear(canvas):
        canvas.delete("all")

    PEN_SIZE = 5

    def predict(canvas):
        img = Image.new('L', (280, 280), color='black')
        draw = ImageDraw.Draw(img)

        for item in canvas.find_all():
            coords = canvas.coords(item)
            for i in range(0, len(coords), 2):
                x, y = coords[i], coords[i + 1]
                draw.ellipse((x - PEN_SIZE, y - PEN_SIZE, x + PEN_SIZE, y + PEN_SIZE), fill='white')

        img = img.resize((28, 28))
        img = img.convert('L')
        img = np.array(img)
        img = img.reshape(1, 1, 28, 28)  # or img = img.reshape(1, 784) if you use simple nn (no cnn)
        img = img.astype('float32') / 255
        prediction = model.predict(img)
        prediction = np.argmax(prediction)
        label.configure(text="Prediction: " + str(prediction))

    def paint(event):
        x1, y1 = (event.x - PEN_SIZE), (event.y - PEN_SIZE)
        x2, y2 = (event.x + PEN_SIZE), (event.y + PEN_SIZE)
        canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        predict(canvas)  # Call predict function here for real-time prediction

    canvas.bind("<B1-Motion>", paint)

    window.mainloop()


if __name__ == '__main__':
    main()
