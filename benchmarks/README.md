## Бенчмарки

Для оценки скорости сходимости и времени работы мы оценили работу нашей модели на двух датасетах: mnist (попроще) и cifar (чуть сложнее, тут уже добавили конволюции).

Мы сравнивали наш фреймворк с keras (на одинаковых данных и архитектурах).

К сожалению, по производительности мы сильно проигрываем keras, поэтому совсем сложные архитектуры сравнить не получилось.

### mnist

На этом датасете наша модель неплохо сходится, ей даже повезло обыграть keras на меленьком количестве эпох:

![Results screenshot](/benchmarks/benchmark-results/mnist/metrics.png)

По производительности мы проигрываем примерно в 2 раза, что в целом не так плохо, если соотносить время существования keras и tensorflow с нашим проектом :)

![Results screenshot](/benchmarks/benchmark-results/mnist/fitting-time.png)

Попробуем подробнее посмотреть на то, куда же уходит время, отдельно запустив обучение со специальным колбеком - PerformanceMetricsCallback:

```
Metrics report: 
    Full epoch          : total duration=41579ms, average duration=4157ms, last duration=4221ms
    Full batch          : total duration=29541ms, average duration=1ms
    Gradient calculation: total duration=18ms, average duration=0ms
    Forward pass(total times)
            7680ms: Dense[785 x 100 (including bias)]
            1363ms: SigmoidActivation
             575ms: SoftmaxActivation
             424ms: Dense[101 x 10 (including bias)]
    Backward pass(total times)
            7762ms: Dense[785 x 100 (including bias)]
            1271ms: SigmoidActivation
             425ms: Dense[101 x 10 (including bias)]
             423ms: SoftmaxActivation
    Gradient step(total times)
            8178ms: Dense[785 x 100 (including bias)]
             123ms: Dense[101 x 10 (including bias)]
               0ms: SigmoidActivation
               0ms: SoftmaxActivation
    Apply gradients(total times)
             535ms: Dense[785 x 100 (including bias)]
              20ms: Dense[101 x 10 (including bias)]
               0ms: SigmoidActivation
               0ms: SoftmaxActivation
```

В отчёте выше приведена статистика суммарного времени, затраченного на разные этапы обучения сети. \
В сумме выходит `28779ms` - это практически равно `Full batch total duration=29541ms` - суммарному времени, потраченному непосредственно на обучение на батчах.\
При этом суммарное время всех эпох (`Full epoch: total duration=41579ms`) больше примерно на 30% - где-то потратили 10 секунд. Дело тут в том, что помимо непосредственно обучения время уходит на генерацию батчей и на оценку модели после каждой эпохи.

К сожалению, даже на такой простой архитектуре мы отстаём от keras. Можно было бы попробовать сэкономить время на предварительной подготовке данных и оценке модели (выше мы видели, что на это уходит много времени). \
Непосредственно вычисления вряд ли удастся сильно ускорить. Наверняка keras тут имеет большое преимущество за счет GradientTape, который, вероятно, позволяет ему на такой простой сети проводить вычисления сразу по всем слоям: снижаются накладные расходы.

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

На удивление скор после 20 эпох у нас получился даже лучше, чем у keras. Возможно, он бы сошёлся после большего числа эпох, а может какие-то оптимизации производительности в нём влияют на качество. \
Важно оказалось использовать хорошую инициализацию весов, без неё у нас модель сходилась гораздо медленее. Мы в итоге взяли GlorotUniform - такой же, как в керасе.

![Results screenshot](/benchmarks/benchmark-results/cifar/metrics.png)

Однако производительность даёт совсем мало пространства для увеличения числа эпох...

![Results screenshot](/benchmarks/benchmark-results/cifar/fitting-time.png)

Здесь keras далеко впереди, быстрее он примерно в 10 раз. Скорее всего за счёт того, что нам пришлось самостоятельно реализовывать операцию конволюции двух тензоров, так как судя по документации Armadillo нативная реализация конволюции в библиотеке сырая и неоптимизированная, а также она не совсем подходит для нашей архитектуры - легко было бы запутаться.

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

Для запуска с PerformanceMetricsCallback, нужно выполнить ```./run.sh  --perf-callback 1```, так же находясь в **директории ```benchmarks```**.

Для работы бенчмарков потребуется docker.

Внутри будет запущена сначала модель в keras-е, а затем - в нашем фреймворке.
После этого будут построены графики и сохранены в ```benchmarks/benchmark-results``` (они уже лежат там и на этой странице).

Сами тесты описаны в директории ```benchmarks/benchmark-results/cases/<testname>/config.<hpp|py>```:

[mnist keras](./cases/mnist/config.py) \
[mnist nn-framework](./cases/mnist/config.hpp) \
[cifar keras](./cases/cifar/config.py) \
[cifar nn-framework](./cases/cifar/config.hpp)


