from flyai.dataset import Dataset
from model import Model

data = Dataset()
model = Model(data)
x_test = [{'image_path': 'images/750647014.jpg'}]
p = model.predict_all(x_test)
print(p)