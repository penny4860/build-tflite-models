

## Freeze tensorflow graph

* tensorflow model ����
	* model.ckpt
	* graph.pb

* freeze_graph.py �� ����
	python C:\Anaconda3\envs\yolov3\Lib\site-packages\tensorflow\python\tools\freeze_graph.py --input_graph models\graph.pb --input_checkpoint models\model.ckpt --output_graph models/frozen_graph.pb --output_node_names softmax
