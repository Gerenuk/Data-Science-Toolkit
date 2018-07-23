import glob
from PIL import Image
import itertools as itoo
from operator import attrgetter

fileglob=r"D:\Download\train\train\acantharia_protist_halo\*.jpg"
output_file=r"D:\output.jpg"
max_width=800

class OrientedImage:
    _id_gen=itoo.count()

    def __init__(self, image):
        self._image=image
        self.x, self.y=image.size
        self._id=next(OrientedImage._id_gen)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self._id==other._id

    @property
    def image(self):
        x,y=self._image.size
        if (x,y)==(self.x, self.y):
            result=self._image
        else:
            result=self._image.transpose(Image.ROTATE_90)
            assert result.size==(self.x, self.y)
        return result

    def make_largest(self, axis):
        if (axis=="x" and self.x<self.y) or (axis=="y" and self.y<self.x):
            self.x, self.y=self.y, self.x
        return self

class ImageRows:
    def __init__(self, image_S):
        self._image_S=image_S
        self.image_row_LL=[]
        self._current_row_L=[]
        self.cur_row_width=0

    def has_images(self):
        return bool(self._image_S)

    def remaining_images(self):
        for image in self._image_S:
            yield image

    def add_image(self, image):
        self._image_S.remove(image)
        self._current_row_L.append(image)
        self.cur_row_width+=image.x

    def flush_row(self):
        self.image_row_LL.append(self._current_row_L)
        self._current_row_L=[]
        self.cur_row_width=0

    def max_size(self):
        return (max(sum(image.x for image in row) for row in self.image_row_LL),
                sum(max(image.y for image in row) for row in self.image_row_LL))

    def placed_images(self):
        y=0
        for row in self.image_row_LL:
            x=0
            y_L=[]
            row_height=max(image.y for image in row)
            for image in row:
                yield (x,y+(row_height-image.y)//2,image.image)
                x+=image.x
                y_L.append(image.y)
            y+=max(y_L)

image_rows=ImageRows({OrientedImage(Image.open(filename)) for filename in glob.glob(fileglob)})
while image_rows.has_images():
    first_image=max([image.make_largest("x") for image in image_rows.remaining_images()], key=attrgetter("y"))
    image_rows.add_image(first_image)
    row_height=first_image.y

    next_images=[]
    for image in image_rows.remaining_images():
        image.make_largest("y")
        if image.y<=row_height:
            next_images.append(image)
        else:
            image.make_largest("x")
            next_images.append(image)
    next_images=sorted(next_images, key=attrgetter("y"), reverse=True)

    for image in next_images:
        if image_rows.cur_row_width+image.x<=max_width:
            assert image.y<=row_height
            image_rows.add_image(image)

    image_rows.flush_row()

width, height=image_rows.max_size()
print("Total size: {} x {}".format(width, height))
output_image=Image.new("L",(width, height), "white")
for x, y, image in image_rows.placed_images():
    output_image.paste(image,(x, y))
output_image.save(output_file)