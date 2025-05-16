import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def plus_mas(masive, ind):
  sum = 0
  for i in range(len(mas)):
    masive.append(mas[i][ind])
    sum += float(mas[i][ind])
  std_x = sum/len(masive)
  sum = 0
  for i in range(len(dead)):
    sum += float(dead[i])
  std_y = sum/len(dead)
  sum = 0
  for i in range(len(masive)):
    sum += (masive[i] - std_x)**2
  q_x = np.sqrt(sum/(len(masive) - 1))
  sum = 0
  for i in range(len(dead)):
    sum += (dead[i] - std_y)**2
  q_y = np.sqrt(sum/(len(dead) - 1))
  sum = 0
  for i in range(len(masive)):
    sum += ((masive[i] - std_x) / q_x) * ((dead[i] - std_y) / q_y)
  r = sum / (len(masive) - 1)
  print('Коэффициент корреляции:', r)
  plot.scatter(masive, dead)
  plot.show()
  pass
df = pd.read_csv('drive/MyDrive/Colab Notebooks/Firer/total.csv', sep = ';')
mas = np.array(df)
for i in range(len(mas)):
  new = int(mas[i][0].split('.')[2]) + int(mas[i][0].split('.')[0]) + int(mas[i][0].split('.')[1])
  mas[i][0] = new
dead = []
for i in range(len(mas)):
  dead.append(mas[i][11])
f5 = []
print('f5')
plus_mas(f5, 0)
#here
f6 = []
print('f6')
plus_mas(f6, 1)
#here
f7 = []
print('f7')
plus_mas(f7, 2)
#here
f8 = []
print('f8')
plus_mas(f8, 3)
#here
f10 = []
print('f10')
plus_mas(f10, 4)
#here
f11 = []
print('f11')
plus_mas(f11, 5)
#here
f12 = []
print('f12')
plus_mas(f12, 6)
#here
f14 = []
print('f14')
plus_mas(f14, 7)
f15 = []
print('f15')
plus_mas(f15, 8)
f16 = []
print('f16')
plus_mas(f16, 9)
#here
f26 = []
print('f26')
plus_mas(f26, 10)
#here
f30 = []
print('f30')
plus_mas(f30, 12)
f36 = []
print('f36')
plus_mas(f36, 13)
f39 = []
print('f39')
plus_mas(f39, 14)
f40 = []
print('f40')
plus_mas(f40, 15)
#here
f43 = []
print('f43')
plus_mas(f43, 16)
f44 = []
print('f44')
plus_mas(f44, 17)
f56 = []
print('f56')
plus_mas(f56, 18)
f75 = []
print('f75')
plus_mas(f75, 19)
#here
f83 = []
print('f83')
plus_mas(f83, 20)
f88 = []
print('f88')
plus_mas(f88, 21)
f100 = []
print('f100')
plus_mas(f100, 22)
f106 = []
print('f106')
plus_mas(f106, 23)
deadmas = []
for i in range(len(mas)):
  if mas[i][11] != 0:
    deadmas.append([mas[i][0], mas[i][1], mas[i][2], mas[i][3], mas[i][4], mas[i][5], mas[i][6], mas[i][9], mas[i][10], mas[i][15], mas[i][19], mas[i][20], mas[i][21], mas[i][22], mas[i][11]])
livemas = []
for i in range(len(mas)):
  if mas[i][11] == 0:
    livemas.append([mas[i][0], mas[i][1], mas[i][2], mas[i][3], mas[i][4], mas[i][5], mas[i][6], mas[i][9], mas[i][10], mas[i][15], mas[i][19], mas[i][20], mas[i][21], mas[i][22], mas[i][11]])
deadX = []
for i in range(len(deadmas)):
  dop = []
  for j in range(len(deadmas[i])):
    if j != 14:
      dop.append(deadmas[i][j])      
  deadX.append(dop)
liveX = []
for i in range(len(livemas)):
  dop = []
  for j in range(len(livemas[i])):
    if j != 14:
      dop.append(livemas[i][j])
  liveX.append(dop)
deady = []
for i in range(len(deadmas)):
  for j in range(len(deadmas[i])):
    if j == 14:
      deady.append(deadmas[i][j])
livey = []
for i in range(len(livemas)):
  for j in range(len(livemas[i])):
    if j == 14:
      livey.append(livemas[i][j])
from sklearn.neighbors import KNeighborsClassifier
scores = []
for i in range(1):
  liveX_train, liveX_test, livey_train, livey_test = train_test_split(liveX, livey, test_size=0.1)
  deadX_train  = deadX + liveX_train
  deadX_test = deadX + liveX_test
  deady_train = deady + livey_train
  deady_test = deady + livey_test
  reg = KNeighborsClassifier(n_neighbors = 1)
  reg.fit(deadX_train, deady_train)
  score = 0
  for j in range(len(deadX_test)):
    if reg.predict([deadX_test[j]]) == deady_test[j]:
      score += 1
  scores.append(score/len(deady_test))
print('Процент угадывания:', np.mean(scores)*100)
a = int(input('Число: '))
b = int(input('Вид населенного пункта: '))
c = int(input('Вид охраны пожурной части пункта: '))
d = int(input('Организационно-правовая форма: '))
e = int(input('Ведомственная принадлежность: '))
f = int(input('Тип предприятия, организации, учреждения: '))
g = int(input('Объект пожара: '))
h = int(input('Степень огнестойкости: '))
i = int(input('Расстояние до пожарной части, км: '))
j = int(input('Единиц: '))
k = int(input('Участники тушения пожара: '))
l = int(input('Количество техники, ед: '))
m = int(input('Подано пожарных стволов: '))
n = int(input('Вдодисточники: '))
result = [a, b, c, d, e, f, g, h, i, j, k, l, m, n]
print('Вероятнее всего погибло:', int(reg.predict([result])[0]))