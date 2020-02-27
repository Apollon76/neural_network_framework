### Как все установить

- Установить [LAPACK](http://www.netlib.org/lapack/)
    * Я собрал его из исходников, но говорят что можно поставить .deb пакет
        Healthcheck: 
            ```
                $> whereis liblapack
                $> liblapack: /usr/local/lib/liblapack.a
            ```
- Установить [OpenBLAS](https://www.openblas.net/)
    * Я тоже собрал его из исходников
        Healthcheck:
            ```
                $> whereis libopenblas
                $> liblapack: /usr/local/lib/libopenblas.a
            ```
- Установить [Armadillo](https://gitlab.com/conradsnicta/armadillo-code)
    * Собрал из исходников, но говорят что есть пакет (хотя упоминают, что он отстает от релизов на гитлабе)
        Healthcheck:
            ```
                $> whereis libarmadillo
                $> libarmadillo: /usr/lib/x86_64-linux-gnu/libarmadillo.so
            ```
