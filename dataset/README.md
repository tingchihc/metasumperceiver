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

### Prompt the claims from the Llama2  

1. follow [llama](https://github.com/meta-llama/llama) method to download in your side. We are using llama2 7B to prompt our claims.
2. Run the following code and then you will get the news claims from each cluster.
```
python prompt_claims.py
```
3. You also can try on this [space](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat). This is how our claims are generated.  

### Reference  

1. [googlesearch](https://github.com/Nv7-GitHub/googlesearch)  
2. [llama](https://github.com/meta-llama/llama)  
