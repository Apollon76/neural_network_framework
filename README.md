# Neural network framework
A study project inspired by [this desciption](https://github.com/yandexdataschool/lsml-projects/blob/master/nn.md).

# How to run

The easiest way to build, run and develop is by using docker.
Make sure that docker has enough RAM (8GB recommended). 

## Unarchive mnist dataset
```
unzip data/kaggle-digit-recognizer/digit-recognizer.zip -d data/kaggle-digit-recognizer/ 
```

## Run main
```
docker-compose build
docker-compose run -v $(pwd):/nn_framework nn_framework run -d /nn_framework
```

## Run benchmark
```
cd benchmarks && ./run.sh
```

## Run docker for local developement in JetBrains Clion:
```
docker-compose up --build
```

For remote toolchain setup follow the instructions on this link: https://blog.jetbrains.com/clion/2018/09/initial-remote-dev-support-clion/

```
ssh port: 2223
ssh user: user
ssh password: password
```
