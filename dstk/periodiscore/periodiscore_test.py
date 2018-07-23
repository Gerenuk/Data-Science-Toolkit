from functools import partial
from pprint import pprint

from individual.anton.periodiscore.periodi_bin import PeriodiBin
from individual.anton.periodiscore.periodi_train import PeriodiTrain
from individual.anton.periodiscore.periodi_predict import PeriodiPredict
from individual.anton.periodiscore.periodi_scorer import PeriodiScorer


def bin_keys_func(point, width):
    return [(point % width, point)]


def bin_generator(bin_key, width):
    return PeriodiBin(lambda point: (1, point // width), name=str(bin_key))


data = [11, 35, 51, 72]

data1 = [11, 35]
data2 = [51, 72]

periodi_train1 = PeriodiTrain(partial(bin_generator, width=10),
                              partial(bin_keys_func, width=10))
for data_x in data1:
    periodi_train1.add(data_x)

periodi_train2 = PeriodiTrain(partial(bin_generator, width=10),
                              partial(bin_keys_func, width=10))
for data_x in data2:
    periodi_train2.add(data_x)

periodi_train = periodi_train1.merge(periodi_train2)

pprint(periodi_train.bin_dict)

periodi_predict = PeriodiPredict(periodi_train, PeriodiScorer, lambda bin: [bin])
print("PeriodiPredict:", periodi_predict)

print("Data scores:")
for data_x in data:
    print(data_x, "->", periodi_predict.score(data_x))

print("--- Top bins ---")
for top_bin in periodi_predict.get_top():
    print(top_bin)
