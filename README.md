# Neural network framework
A study project inspired by [this desciption](https://github.com/yandexdataschool/lsml-projects/blob/master/nn.md).

# How to run

The easiest way to build, run and develop is by using docker.

## Run main
```
docker-compose build
docker-compose run -v $(pwd):/nn_framework run
```

## Run docker for local developement in JetBrains Clion:
```
docker-compose up --build development
```

For remote toolchain setup follow the instructions on this link: https://blog.jetbrains.com/clion/2018/09/initial-remote-dev-support-clion/

```
ssh port: 2223
ssh user: user
ssh password: password
```
