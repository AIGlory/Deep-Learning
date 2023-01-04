# Deep-Learning
(Assignment of course Neural Network and Deep Learning(TJU))
实验环境
pytorch,安装包：torch,torch.nn,torchvision,torch.utils.data,torchvision.transforms
数据集下载
MNIST数据集，下载链接http://yann.lecun.com/exdb/mnist/
运行方式
在各编程平台上直接编译运行即可，如pycharm中Alt+Shift+F10
实验结果
model	epoch(s)	batch_size	learning_rate	accuracy
RNN	1	64	0.01	0.5805
RNN	1	256	0.01	0.827
RNN	1	192	0.01	0.8681
RNN	4	192	0.01	0.8802
RNN	5	192	0.01	0.7648
RNN	6	192	0.01	0.5635
RNN	1	192	0.001	0.7682
RNN	1	192	0.1	0.1032
LSTM	1	64	0.01	0.9668
LSTM	1	256	0.01	0.9585
LSTM	1	512	0.01	0.9141
LSTM	1	6000	0.01	0.4406
LSTM	10	6000	0.01	0.956
LSTM	10	64	0.01	0.9793
LSTM	20	64	0.01	0.9795
LSTM	50	64	0.01	0.7545
LSTM	1	64	0.001	0.9057
LSTM	1	64	0.1	0.7715
GRU	1	64	0.01	0.9657
GRU	1	256	0.01	0.9581
GRU	1	512	0.01	0.9245
GRU	10	64	0.01	0.9487
GRU	20	64	0.01	0.9657
GRU	50	64	0.01	0.9707
GRU	1	64	0.001	0.9187
GRU	1	64	0.1	0.6343
