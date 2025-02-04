�	f���8T>@f���8T>@!f���8T>@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$f���8T>@v5y�*�?1o�j�;@I�+����?:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*moderate2A5.6 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	v5y�*�?v5y�*�?!v5y�*�?      ��!       "	o�j�;@o�j�;@!o�j�;@*      ��!       2      ��!       :	�+����?�+����?!�+����?B      ��!       J      ��!       R      ��!       Z      ��!       JGPU�"]
4gradients/conv2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput� ³t˵?!� ³t˵?"%

input_1/_2_Recv���E�?!Xp_�E��?"_
5gradients/conv1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter6�x�Ա?!��Hr�?"<
gradients/variance_grad/truedivRealDiv�o�����?!� ƙ���?"8
SquaredDifferenceSquaredDifference��Ko�f�?!�����e�?"]
@gradients/AddN_7-1-TransposeNHWCToNCHW-LayoutOptimizer:TransposeUnknown��
��?!D��}T��?"-
conv2/convolutionConv2D�>H;��?!���[�?"_
5gradients/conv2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterv�7d�?!~�J���?"]
4gradients/conv3/convolution_grad/Conv2DBackpropInputConv2DBackpropInputP^w栜?!~������?"D
+gradients/bn_conv1/batchnorm/mul_1_grad/MulMul��l����?!�I�i�h�?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nomoderate"A5.6 % of the total step time sampled is spent on All Others time.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 