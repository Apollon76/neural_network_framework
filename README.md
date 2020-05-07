# Neural network framework
Фреймворк для обучения нейронных сетекй - учебный проект в рамках [курса шада](https://github.com/yandexdataschool/lsml-projects/blob/master/nn.md).

## О проекте

Проект написан на c++ с использованием библиотеки Armadillo.

## Структура проекта

```docker``` - описание docker-образа со всеми зависимостями

```src``` - исходный код проекта

```tests``` - тесты

```examples``` - простые примеры использования библиотеки и описания к ним

```examples``` - бенчмарки и мехака для их запуска

```data``` - данные или скрипты для их получения, которые используется в примерах

## Знакомство с фрейморком

Для работы с проектом потребуется docker и docker-compose. \
Мы рекомендуем дать докеру хотя бы 8GB оперативной памяти.

#### Примеры
Рекомендуем посмотреть на примеры работы с фреймворком в папке ```examples``` (там же есть более подробный [readme](../blob/master/examples/README.md)) о них).

#### Бенчмарки
Также рекомендуем взглянуть на [результаты бенчмарков](../blob/master/benchmarks/README.md).

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