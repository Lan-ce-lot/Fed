python main.py -nc 5 -jr 1 -nb 10 -data digits -m cnn -algo FedBN

python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -algo FedBN
python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -algo FedAvg
python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -algo Fed

for pickle cifar100
python main.py -nc 5 -jr 1 -nb 20  -lr 0.001 -lbs 40 -data Cifar -m cnn -algo Fed
python main.py -nc 5 -jr 1 -nb 20  -lr 0.001 -lbs 40 -data Cifar -m cnn -algo FedBN
python main.py -nc 5 -jr 1 -nb 20  -lr 0.001 -lbs 40 -data Cifar -m cnn -algo FedAvg

for cifar20
python main.py -nc 50 -jr 1 -nb 20  -lr 0.01 -lbs 40 -data Cifar -m cnn -algo Fed
python main.py -nc 50 -jr 1 -nb 20  -lr 0.01 -lbs 40 -data Cifar -m cnn -algo FedBN
python main.py -nc 50 -jr 1 -nb 20  -lr 0.01 -lbs 40 -data Cifar -m cnn -algo FedAvg