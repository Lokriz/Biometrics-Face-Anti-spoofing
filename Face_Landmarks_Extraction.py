import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import dlib
import math
import statistics
from moviepy.editor import *


# Функция поиска гауссовой площади по коодинатам точек многоугольника
def PolyArea(x, y, nl):
    norm = []
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    for i in range(len(nl) - 1):
        norm.append(math.sqrt((nl[i+1][0] - nl[i][0]) ** 2 + (nl[i+1][1] - nl[i][1]) ** 2))
    poly = area * 4 / (norm[0]**3 / norm[1])
    print(norm[0])
    return round(poly, 3)

angle_types = ['a_0', 'a_30(1)', 'a_30(2)', 'b_0', 'b_30(1)', 'b_30(2)']
angle_types = ['b_30(2)']

for angle_type in angle_types:

    name = 'Ilyas10/' + angle_type
    cap = cv2.VideoCapture('D:/USR/Рабочий стол/НИР(3сем)/Records/' + name + '.MOV')

    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Обозначение вспомогательных переменных
    counter = 0
    area_lst = []
    vec = []
    poly_area = 0

    # Вызов видео-записи
    while True:
        success, frame = cap.read()

        if success:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Детекция лиц на потоке изображений
            faces = detector(gray)
            counter += 1

            for face in faces:

                # Поиск изображения лиц на исходной видеозаписи
                landmarks = predictor(gray, face)

                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                poly_x, poly_y, norm_val = [], [], []

                # Размещение маркеров на изображении лица
                for n in [39, 42, 30] + list(range(60, 68)):
                    x_mark = landmarks.part(n).x
                    y_mark = landmarks.part(n).y
                    if n < 60:
                        norm_val.append((x_mark, y_mark))
                        cv2.circle(frame, (x_mark, y_mark), 2, (255, 0, 255), -1)
                    else:
                        poly_x.append(x_mark)
                        poly_y.append(y_mark)
                        cv2.circle(frame, (x_mark, y_mark), 2, (0, 255, 0), -1)
                    vec.extend([x_mark, y_mark])

                # Подсчет площади, ограниченной внутренней частью губ
                poly_area = PolyArea(poly_x, poly_y, norm_val)
                vec.append(poly_area)
            area_lst.append(poly_area)
            #print(poly_area)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    vec_array = np.array(vec).reshape(-1, 23)
    print(vec_array, len(vec))


    # Отсечение значений без активности
    lst_np = np.array(area_lst)
    if statistics.mode(area_lst) < 0.08:
        lst_mode = statistics.mode(lst_np[lst_np > statistics.mode(area_lst)])
    else:
        lst_mode = statistics.mode(area_lst)

    level_of_cut = lst_mode*1.6 + statistics.mean(area_lst) * 0.5

    print(lst_mode, level_of_cut)

    start_time = []
    fin_time = []
    start, fin = [], []

    # Поиск стартовых и конечных точек активности
    for j in range(len(area_lst)):
        if all([i > level_of_cut for i in area_lst[j - 4: j]]) and j > 10 and len(start_time) <= len(fin_time) \
                and any([i > level_of_cut*1.4 for i in area_lst[j - 4: j]]):
            start_time.append(round((j - 7) / fps, 3))
            start.append(j-7)

        elif round(area_lst[j], 2) <= level_of_cut and all([round(i, 2) <= level_of_cut for i in area_lst[j - 6: j]]) \
                and len(start_time) > len(fin_time):
            fin_time.append(round((j - 4) / fps, 3))
            fin.append(j-4)
        if len(fin) == 10:
            break

    print(start_time)
    print(fin_time)
    print(start)
    print(fin)

    # Построение графика изменения площади со стартовыми и конечными точками артикуляционной активности
    plt.figure(figsize=(13, 9))
    time = np.arange(counter) / fps
    plt.plot(time, area_lst, label='_nolegend_')
    plt.title('Изменение площади', y=1.04, fontsize=20)
    plt.xlabel('Время, сек', fontsize=15, labelpad=10)
    plt.ylabel('Нормированная площадь', fontsize=16, labelpad=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    try:
        for i in range(len(start_time)):
            plt.axvline(x=start_time[i], color='g', alpha=0.5)
            plt.axvline(x=fin_time[i], color='r', alpha=0.5)
        plt.legend(['Нач. записи', 'Конец записи'], prop={'size': 14})
    except Exception:
        print('Не удалось разбить')
    plt.show()


    def conv(x):
        for i in range(len(x)):
            if x[i] == 0: x[i] = 'zero'
            elif x[i] == 1: x[i] = 'one'
            elif x[i] == 2: x[i] = 'two'
            elif x[i] == 3: x[i] = 'three'
            elif x[i] == 4: x[i] = 'four'
            elif x[i] == 5: x[i] = 'five'
            elif x[i] == 6: x[i] = 'six'
            elif x[i] == 7: x[i] = 'seven'
            elif x[i] == 8: x[i] = 'eight'
            elif x[i] == 9: x[i] = 'nine'
        return x

    value = 'not_num'
    mark_types_a = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    mark_types_b = conv([5, 1, 9, 6, 3, 7, 0, 4, 8, 2])
    mark = []
    k = 0

    for i in range(0, counter):
        if any([j == i for j in start]):
            value = mark_types_a[k]
            k += 1
        if any([j == i for j in fin]):
            value = 'not_num'
        mark.append(value)

    columns = ['x39', 'y39', 'x42', 'y42', 'x30', 'y30', 'x60', 'y60', 'x61', 'y61', 'x62', 'y62', \
    'x63', 'y63', 'x64', 'y64', 'x65', 'y65', 'x66', 'y66', 'x67', 'y67', 'square']
    df = pd.DataFrame(vec_array, columns=columns)
    df['target'] = mark
    if len(df['target'].unique()) == 11:
        df.to_csv(path_or_buf=name + '.csv')
    else:
        print('Not enough marks')

# Разделение исходного видео на фрагменты по найденным тайм-кодам
# for i in range(len(start_time)):
#     clip = VideoFileClip(name).subclip(start_time[i], fin_time[i])
#     clip.write_videofile('test' + str(i) + '.mp4', audio= False)

