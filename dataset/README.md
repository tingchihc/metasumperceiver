## Dataset  

### Preprocessing  

1. Download the Multi-News dataset from Huggingface on your side. [link](https://huggingface.co/datasets/alexfabbri/multi_news)
2. Process multi_news data with [documents.json](https://drive.google.com/file/d/1dE_0UmfDH5XggrLWD6xRK09BWIXjRTmc/view?usp=sharing) format or download this directly:

```
# '0_0' represents a unique document identifier:  
# The first '0' is the cluster number from the Multi_news dataset (group of documents).  
# The second '0' is the document ID within that cluster.  
{
'train': {'0_0': '', '0_1': ''}, ...
'validation': {'0_0': ''}, ...
'test': {'0_0': ''}, ...
}
```

### Search news images  

1. Run the following code and you will get the news images similar like the [sample](https://github.com/tingchihc/metasumperceiver/tree/main/dataset/sample).  
```
python search_img.py  
```

### Reference  

1. [googlesearch](https://github.com/Nv7-GitHub/googlesearch)  
