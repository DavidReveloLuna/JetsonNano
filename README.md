# JetsonNano

## 1. Instalación Jetpack. Configuración inicial

[Tutorial #1 - click aqui](https://www.youtube.com/watch?v=6uqM6ltCLlE&list=PLsjK_a5MFguIUJJ1GPt1I2eN6cihKg2kG)


## 2. Detección de objetos Hello World desde Docker

[Tutorial #2 - click aqui](https://www.youtube.com/watch?v=6uqM6ltCLlE&list=PLsjK_a5MFguIUJJ1GPt1I2eN6cihKg2kG)

    $ git clone --recursive https://github.com/dusty-nv/jetson-in...
    $ cd jetson-inference
    $ docker/run.sh
    $ cd build/aarch64/bin

Para ejecutar los demos en la carpeta bin:

    $ ./video-viewer /dev/video0
    $ ./segnet /dev/video0
    $ ./detectnet /dev/video0
    $ ./depthnet /dev/video0
    $ ./posenet /dev/video0
    
## 3. Entrenamiento de SSD-Mobilenet

### Configuración de memoria swap

    $ sudo systemctl disable nvzramconfig
    $ sudo fallocate -l 4G /mnt/4GB.swap
    $ sudo mkswap /mnt/4GB.swap
    $ sudo swapon /mnt/4GB.swap

Agregar la siguiente linea al final del archivo /etc/fstab para que los cambios se guarden permanentemente

    /mnt/4GB.swap  none  swap  sw 0  0

### Descarga de imagenes para entrenamiento

    $ python3 open_images_downloader.py --max-images=2500 --class-names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon" --data=data/fruit
    
### Entrenamiento de la red

    $ python3 open_images_downloader.py --max-images=2500 --class-names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon" --data=data/fruit

### Convertir modelo a formato ONNX para TensorRT

    $ python3 onnx_export.py --model-dir=models/fruit

### Prueba del modelo en imagenes

    $ detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt \
              --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
                "/jetson-inference/python/training/detection/ssd/data/fruit/test/*.jpg" /jetson-inference/data/images/test/fruit_%i.jpg

### Prueba del modelo en tiempo real con webcam

    $ detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt \
              --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
                /dev/video0
