# Neural network framework
Фреймворк для обучения нейронных сетекй - учебный проект в рамках [курса шада](https://github.com/yandexdataschool/lsml-projects/blob/master/nn.md).

## О проекте

Проект написан на c++ с использованием библиотеки Armadillo.

## Структура проекта

```docker``` - описание docker-образа со всеми зависимостями

```src``` - исходный код проекта

```tests``` - тесты

```examples``` - простые примеры использования библиотеки и описания к ним

```benchmarks``` - бенчмарки и механика для их запуска

```data``` - данные или скрипты для их получения, которые используется в примерах

## Знакомство с фрейморком

Для работы с проектом потребуется docker и docker-compose. \
Мы рекомендуем дать докеру хотя бы 8GB оперативной памяти.

#### Примеры
Рекомендуем посмотреть на примеры работы с фреймворком в папке ```examples``` (там же есть более подробный [readme](../master/examples/README.md) о них).

Пример кода простейшей нейронной сети:
```C++
auto neural_network = NeuralNetwork<double>(
        std::make_unique<RMSPropOptimizer<double>>(0.01),
        std::make_unique<MSELoss<double>>()
);
neural_network
        .AddLayer(std::make_unique<DenseLayer<double>>(10, 1))
        .AddLayer(std::make_unique<SoftmaxActivationLayer<double>>());
Tensor<double> input = Tensor<double>::init(/* your innput tensor here */);
Tensor<double> output = Tensor<double>::init(/* your output tensor here */);
neural_network.Fit(input, output, 10); // fit model with 10 epochs
```

#### Бенчмарки
Также рекомендуем взглянуть на [результаты бенчмарков](../master/benchmarks/README.md).

#### Тесты
Для запуска тестов можно выполнить следующую команду:

```
docker-compose run -v $(pwd):/nn_framework nn_framework test
```

## Разработка

#### JetBrains Clion 
Для разработки в Clion удобнее всего использовать remote toolchain в докере:
```
docker-compose up --build
```

Инструкции по её настройке есть [тут](https://blog.jetbrains.com/clion/2018/09/initial-remote-dev-support-clion/).

```
ssh port: 2223
ssh user: user
ssh password: password
```

#### Запуск main в докере
```
docker-compose build
docker-compose run -v $(pwd):/nn_framework nn_framework run -d /nn_framework
```

#### Запуск бенчмарков
```
cd benchmarks && ./run.sh
```
