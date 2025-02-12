from data.data import Data
from model.model import Model


if __name__ == '__main__':
    data = Data()
    data.extract_zip()
    data.describe()
    data.prepare_data()

    model = Model()
    model.train(data.x, data.y)

    data.clean()
