## Бенчмарки

Для оценки скорости сходимости и времени работы мы оценили работу нашей модели на двух датасетах: mnist (попроще) и cifar (чуть сложнее, тут уже добавили конволюции).

Мы сравнивали наш фреймворк с keras (на одинаковых данных и архитектурах).

К сожалению, по производительности мы сильно проигрываем keras, поэтому совсем сложные архитектуры сравнить не получилось.

### mnist

На этом датасете наша модель неплохо сходится, ей даже повезло обыграть keras на меленьком количестве эпох:

![Results screenshot](/benchmarks/benchmark-results/mnist/metrics.png)

По производительности мы проигрываем примерно в 3 раза, что в целом не так плохо, если соотносить время существования keras и tensorflow с нашим проектом :)

![Results screenshot](/benchmarks/benchmark-results/mnist/fitting-time.png)

Архитектура модели была такая (в плюсах такая же):

```
model = Sequential([
    L.Input(784),
    L.Dense(100, activation='sigmoid'),
    L.Dense(10, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
```


### cifar

Здесь результаты уже не такие хорошие. Модель постепенно сходится, так что, возможно, при большем числе эпох она бы и догнала keras.

![Results screenshot](/benchmarks/benchmark-results/cifar/metrics.png)

Однако производительность даёт совсем мало пространства для увеличения числа эпох...

![Results screenshot](/benchmarks/benchmark-results/cifar/fitting-time.png)

Здесь keras далеко впереди. Скорее всего - из-за самописной реализации конволюции. Такая была выбрана, т.к. в armadillo реализация, судя по документациии, тоже сырая, но ещё и не совсем подходит для нашей архитектуры - легко было бы запутаться.

Архитектура модели была такая (в плюсах такая же):

```
model = Sequential([
    L.Conv2D(10, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    L.MaxPooling2D((2, 2)),
    L.Conv2D(5, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    L.MaxPooling2D((2, 2)),
    L.Flatten(),
    L.Dense(100, activation='sigmoid'),
    L.Dense(10, activation='softmax'),
])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
```

### Про механику бенчмарков

Запустить бенмарки можно командой ```./run.sh```, находясь в **директории ```benchmarks```**.

Для работы бенчмарков потребуется docker.

Внутри будут запущена сначала модель в keras-е, а затем - в нашем фреймворке.
После этого будут построены графики и сохранены в ```benchmarks/benchmark-results``` (они уже лежат там и на этой странице).

Сами тесты описаны в директории ```benchmarks/benchmark-results/cases/<testname>/config.<hpp|py>```:

[mnist keras](./cases/mnist/config.py) \
[mnist nn-framework](./cases/mnist/config.hpp) \
[cifar keras](./cases/cifar/config.py) \
[cifar nn-framework](./cases/cifar/config.hpp)


