

## Freeze tensorflow graph

#### 1. Build tensorflow model

* output: model.ckpt, graph.pb

#### 2. Run freeze_graph.py

```python C:\Anaconda3\envs\yolov3\Lib\site-packages\tensorflow\python\tools\freeze_graph.py --input_graph models\graph.pb --input_checkpoint models\model.ckpt --output_graph models/frozen_graph.pb --output_node_names softmax```


#### 3. Run tf-lite converter

```tflite_convert --graph_def_file=models/frozen_graph.pb --output_file=models/quantized_graph.lite --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --input_shape=1,784 --input_array=input --output_array=softmax --inference_type=QUANTIZED_UINT8  --default_ranges_min=0 --default_ranges_max=6 --mean_values=128 --std_dev_values=127```

