docker build -t mlservice .

docker run -d -p 56733:80 --name=mltest -v $PWD:/app mlservice     

sudo docker stop e78a && sudo docker start e78a


docker run -d -p 56733:80 --name=mltest -v E:\Lakna\Flask\mlservice:/app mlservice  