# AWDNN
An Attention-based Wide and Deep Neural Network for Reentrancy Vulnerability Detection in Smart Contracts

## Introduction
This paper presents an Attention-based Wide and Deep Neural Network (AWDNN) for Reentrancy vulnerability Detection in Ethereum smart contracts.
 Our approach includes three phases: code optimization, vectorization, and vulnerability detection. 

## Requirements

#### The following packages are required to run WIDENNET
* **python**3.0
* **Tensorflow** 2.9.0
* **sklearn**
* **matplotlib**
* **gensim**
* **solcx**
* **pyevmasm**
* **pandas**
* **numpy**
* **pandas**

## Dataset
We utilized a publicly available smart contract dataset from GitHub, published by the authors of a notable research paper in blockchain security and smart contract analysis. This dataset comprises a comprehensive collection of smart contracts sourced from the Ethereum Platform (over 96%), GitHub repositories, and
blog posts that analyze contracts. The results of our work were compared against the performance metrics published in the same paper that provided
the dataset on GitHub. Link to the dataset: https://github.com/Messi-Q/Smart-Contract-Dataset

 * dataset for reentrancy vulnerability
 `contracts_re.txt`

## Code Files

* this is the main and base class file. It is implemented in python.
`AWDNN.py` 

* AWDNN class file.
  `config\models\wdnn_att.py`

* contains the attention class file.
  `config\attention\custom_attention.py`

* fragment vectorizer python file
  `config\fragment_vectorizer.py` 



## Running Project
* To test AWDNN:
1. setup your environment using the packages in the requirements
2. ensure you have the right dataset in place: `contracts_re.txt` for reentrancy.
3. for vulnerability type (**-vt**): `re` for reentrancy
   
4. For reentrancy, execute the command:
```
  python3 AWDNN.py .\contracts_re.txt -vt re
```
   
