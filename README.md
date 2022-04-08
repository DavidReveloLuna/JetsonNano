# JetsonNano

## 1. Instalación Jetpack. Configuración inicial

[Tutorial Youtube - click aqui](https://www.youtube.com/watch?v=6uqM6ltCLlE&list=PLsjK_a5MFguIUJJ1GPt1I2eN6cihKg2kG)


## 2. Detección de objetos Hello World desde Docker

[Tutorial Youtube - click aqui](https://www.youtube.com/watch?v=6uqM6ltCLlE&list=PLsjK_a5MFguIUJJ1GPt1I2eN6cihKg2kG)

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

[Tutorial Youtube - click aqui](https://www.youtube.com/watch?v=HXFVexBPjMk&list=PLsjK_a5MFguIUJJ1GPt1I2eN6cihKg2kG&index=3)

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

    $ python3 train_ssd.py --data=data/fruit --model-dir=models/fruit --batch-size=4 --epochs=30

### Convertir modelo a formato ONNX para TensorRT

    $ python3 onnx_export.py --model-dir=models/fruit

### Prueba del modelo en imagenes

    $ detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes "/jetson-inference/python/training/detection/ssd/data/fruit/test/*.jpg" /jetson-inference/data/images/test/fruit_%i.jpg

### Prueba del modelo en tiempo real con webcam

    $ detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0

## 4. Entrenamiento para detección de mascarillas + etiquetado de dataset 

[Tutorial Youtube - click aqui](https://www.youtube.com/watch?v=HC8bq3fFoTk&list=PLsjK_a5MFguIUJJ1GPt1I2eN6cihKg2kG&index=5)

### Etiquetado de dataset

    $ camera-capture /dev/video0

### Entrenamiento de la red

    $ python3 train_ssd.py --dataset-type=voc --data=data/Mask/ --model-dir=models/Mask --batch-size=2 --epochs=10
    
### Convertir modelo a formato ONNX para TensorRT

    $ python3 onnx_export.py --model-dir=models/Mask/

### Prueba del modelo en tiempo real con webcam

    $ detectnet --model=models/Mask/ssd-mobilenet.onnx --labels=models/Mask/labels.txt \
              --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
                /dev/video0
## 5. Construye tu propio código para detección de objetos en JetsonNano

[Tutorial Youtube - click aqui](https://colab.research.google.com/drive/1PrzHKE0yKtyGWIlWIC5OAZv4ywMGDDTZ?usp=sharing)

En este punto debes tener el archivo labels.txt y el modelo con extensión .onnx

### Montar el docker con nuestro modelo ya entrenado
    *   Creamos una carpeta en el directorio raiz (fuera del docker)
    *   Copiamos en la carpeta los archivos labels.txt, ssd-mobilenet.onnx y nuestro programa my_detection.py (por ahora este archivo vacío)
    *   Si es necesario dar permisos a la carpeta usando chmod
    *   Montamos el docker agregando esta carpeta
    
    $ sudo mkdir my_project
    $ sudo chmod -R a+rwx my_project
    $ cd jetson-inference
    $ docker/run.sh --volume ~/my_project:/my_project

### Construimos nuestro código (my_detection.py)

    import jetson.inference
    import jetson.utils

    net = jetson.inference.detectNet("ssd-mobilenet-v2",["--model=/my_project/ssd-mobilenet.onnx","--labels=/my_project/labels.txt","--input-blob=input_0","--output-cvg=scores","--output-bbox=boxes"])
    camera = jetson.utils.videoSource("/dev/video0")
    display = jetson.utils.videoOutput("display://0")

    while display.IsStreaming():
        img = camera.Capture()
        detections = net.Detect(img)
        display.Render(img)
        display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

### Prueba de nuestro código

    $ python3 /my_project/my_detection.py
    
## 6. Entrenamiento de modelo SSD en google Colab para Jetson Nano
[Tutorial Youtube - click aqui (Parte 1)](https://www.youtube.com/watch?v=KOcY-Ga0ZSo&list=PLsjK_a5MFguIUJJ1GPt1I2eN6cihKg2kG&index=9)

[Tutorial Youtube - click aqui (Parte 2)](https://www.youtube.com/watch?v=2YVeCy393Kg&list=PLsjK_a5MFguIUJJ1GPt1I2eN6cihKg2kG&index=10)

Usaremos labelImg para etiquetar el dataset, la carpeta TrainingTools.zip y google Colab para entrenar el SSD en el siguiente documento de colab.

[Entrenamiento en Colab](https://colab.research.google.com/drive/1PrzHKE0yKtyGWIlWIC5OAZv4ywMGDDTZ?usp=sharing)

Algunas instrucciones que usamos en consola

    $   docker/run.sh --volume ~/my_project:/my_project
    $   python3 onnx_export.py --model-dir=models/Chess 
    $   detectnet --model=models/Chess/ssd-mobilenet.onnx --labels=models/Chess/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0
    $   python3 /my_project/Chess.py
