
## model description:

### - as proposed in the paper ["Name Nationality Classification with Recurrent Neural Networks"](https://www.ijcai.org/Proceedings/2017/0289.pdf)

### - uni-, bi- and tri-gram represented input name will each pass one of three backbone LSTMs 
### - the output embeddings of the last timestep will get concatenated to one feature-vector
### - this vector will be passed through a Relu dense layer and finally through a softmax classification dense layer
