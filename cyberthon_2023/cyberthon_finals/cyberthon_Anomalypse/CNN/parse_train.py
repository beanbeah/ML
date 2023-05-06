import csv
from PIL import Image

with open('train.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)

    for row in csvreader:
        data = [int(x) for x in row[:-1]]

        img = Image.new('RGB', (20, 20))

        for i in range(len(data)):
            color = data[i]
            img.putpixel((i%20, i//20), (color, color, color))

        img.save('output{}.png'.format(csvreader.line_num - 1))
