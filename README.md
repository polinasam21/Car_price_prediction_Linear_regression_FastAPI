# Car_price_prediction_Linear_regression_FastAPI

В данном задании была обучена модель линейной регрессии для предсказания стоимости автомобиля, был реализован веб-сервис на FastAPI для применения модели на новых данных

## Часть 1 | EDA и визуализация

1. В данных были обнаружены пропуски, они были заполнены медианами из тренировочного датасета
2. Из тренировочного датасета были удалены дубликаты
3. Из признаков, являющихся числовыми, были удалены единицы измерения, они были переведены во float, целочисленные признаки были переведены в int
4. Были посчитаны основные статистики по числовым и категориальным признакам, построены диаграммы распределения числовых признаков, построена таблица корреляций числовых признаков

## Часть 2 | Модель только на вещественных признаках

1. Была обучена классическая линейная регрессия на вещественных признаках, на тестовых данных r2_score = 0.594, MSE = 233298779730
2. Была проведена стандартизация признаков
3. Была обучена Lasso регрессия, на тестовых данных r2_score = 0.594, MSE = 233299450599
4. При помощи GridSearchCV были подобраны оптимальные параметры Lasso регресии, оптимальное значения составило alpha = 1072.267
5. При помощи GridSearchCV были подобраны оптимальные параметры ElasticNet регрессии, оптимальные параметры составили alpha = 0.088586679, l1_ratio = 0.8421

## Часть 3 | Добавляем категориальные фичи

1. Был удален столбец name
2. При помощи OneHot encoding были закодированы категориальные признаки
3. При помощи GridSearchCV были подобраны оптимальные параметры Ridge регрессии, оптимальное значения составило alpha = 613.59
4. Была обучена Ridge регрессия на данных, включающих закодированные OneHot encoding категориальные признаки, на тестовых данных r2_score = 0.612, MSE = 223020782540

## Часть 4 | Бизнесовая

1. Была реализована бизнес-метрика, представляющая собой среди всех предсказанных цен на авто долю прогнозов, отличающихся от реальных цен на эти авто не более, чем на 10%
2. Была посчитана бизнес-метрика для всех обученных моделей. Наилучшее значение бизнес метрики наблюдалось у Ridge регрессии, обученной на данных, включающих закодированные OneHot encoding категориальные признаки, и составило business_metric = 0.255

## Часть 5 | Реализация сервиса на FastAPI 

1. Был реализован пайплайн модели, модель была обучена, веса сохранились в pickle файл
2. На FastAPI был создан веб-сервис, реализующий две функции: на вход в формате json подаются признаки одного объекта, на выходе сервис выдает предсказанную стоимость машины, на вход подается csv-файл с признаками тестовых объектов, на выходе получаеется файл с +1 столбцом - предсказаниями на этих объектах

## Пример работы сервиса при отправки данных в формате JSON:

![img1](https://github.com/user-attachments/assets/e527aeb7-f058-4489-bfd0-d1e74e2860c9)

![img2](https://github.com/user-attachments/assets/fb7cd372-4552-45c2-95a1-0b31a9bee6ba)

## Пример работы сервиса при отправке csv файла:

![img3](https://github.com/user-attachments/assets/a25c0d96-57f5-4c30-ad1f-8ece0fb6834a)

![img4](https://github.com/user-attachments/assets/1fa41499-2c34-4ae2-96c8-5481ab5a80a2)

![img5](https://github.com/user-attachments/assets/3681c722-4f1f-4042-adb1-2650213ddbff)

![img6](https://github.com/user-attachments/assets/f591907f-3690-47fa-8c66-8c4e3c88642e)
